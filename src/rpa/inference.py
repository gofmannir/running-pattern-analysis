"""End-to-end inference on a raw trackside video using a trained VideoMAE model.

Given a raw video and a local model directory, this script:
  1. Loads the trained VideoMAE model
  2. Runs the full preprocessing pipeline (YOLO detection -> runner filtering ->
     foot-region ROI extraction -> temporal slicing into 16-frame clips)
  3. Groups extracted clips by runner track_id
  4. Selects the PRIMARY runner (track with the most clips)
  5. Runs VideoMAE inference on each clip of the primary runner
  6. Aggregates per-clip predictions via mean softmax -> argmax
  7. Writes a JSON report with the runner's predicted running pattern

Model files are expected to be available locally. Downloading checkpoints from a
shared location (e.g. GCS) is handled separately (see Makefile targets).

Usage:
    uv run python -m rpa.inference \
        --model-dir /path/to/trained_model/best_model \
        --video /path/to/raw_video.mp4 \
        --output results.json

    # Keep intermediate clips for debugging
    uv run python -m rpa.inference \
        --model-dir /path/to/model --video raw.mp4 \
        --output out.json --workdir ./debug_workdir --keep-intermediates
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from transformers import VideoMAEForVideoClassification

from rpa.process_runners import PreprocessorConfig, process_video

# ImageNet normalization constants (must match training preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE input requirements
NUM_FRAMES = 16
FRAME_SIZE = 224

# Matches clip filenames produced by process_runners, e.g. "lap_006_ID_3_clip_001.mp4"
CLIP_NAME_RE = re.compile(r".+_ID_(?P<track>\d+)_clip_\d+\.mp4$")


@dataclass
class ClipResult:
    """Per-clip inference result."""

    clip_name: str
    predicted_idx: int
    predicted_label: int | str
    confidence: float
    probabilities: dict[int | str, float]
    probability_vector: list[float]


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to 3-channel grayscale (matches training preprocessing)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


def load_model(
    model_dir: Path,
) -> tuple[VideoMAEForVideoClassification, dict[int, int | str], torch.device]:
    """Load trained VideoMAE model and label mapping from a local directory.

    Returns:
        Tuple of (model, idx_to_label, device).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {device}", device=device)

    logger.info("Loading model from: {path}", path=model_dir)
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    label_path = model_dir / "label_mapping.json"
    if not label_path.exists():
        logger.warning("label_mapping.json not found; using raw class indices")
        idx_to_label: dict[int, int | str] = {i: i for i in range(model.config.num_labels)}
    else:
        with label_path.open() as f:
            mapping = json.load(f)
        idx_to_label = {int(k): v for k, v in mapping["idx_to_label"].items()}

    logger.info("Model ready with {n} classes: {labels}",
                n=len(idx_to_label), labels=list(idx_to_label.values()))
    return model, idx_to_label, device


def load_video(video_path: Path) -> torch.Tensor | None:
    """Load a clip and preprocess into a (1, T, C, H, W) tensor.

    Preprocessing matches training: grayscale replication, 16-frame center sampling,
    ImageNet normalization.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: {path}", path=video_path)
        return None

    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(
                frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR
            )
            frames.append(to_grayscale(frame_resized))
    finally:
        cap.release()

    if not frames:
        logger.error("No frames extracted from {path}", path=video_path)
        return None

    n_frames = len(frames)
    if n_frames == NUM_FRAMES:
        frames_array = np.stack(frames)
    elif n_frames > NUM_FRAMES:
        start = (n_frames - NUM_FRAMES) // 2
        frames_array = np.stack(frames[start : start + NUM_FRAMES])
    else:
        indices = [i % n_frames for i in range(NUM_FRAMES)]
        frames_array = np.stack([frames[i] for i in indices])

    normalized = frames_array.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # (T, H, W, C) -> (T, C, H, W) -> (1, T, C, H, W)
    tensor = torch.from_numpy(normalized.astype(np.float32)).permute(0, 3, 1, 2)
    return tensor.unsqueeze(0)


def predict_clip(
    model: VideoMAEForVideoClassification,
    clip_path: Path,
    idx_to_label: dict[int, int | str],
    device: torch.device,
) -> ClipResult | None:
    """Run VideoMAE inference on one clip and return per-class probabilities."""
    pixel_values = load_video(clip_path)
    if pixel_values is None:
        return None
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())

    prob_list = [float(p) for p in probs.tolist()]
    probabilities = {idx_to_label[i]: prob_list[i] * 100 for i in range(len(prob_list))}

    return ClipResult(
        clip_name=clip_path.name,
        predicted_idx=pred_idx,
        predicted_label=idx_to_label[pred_idx],
        confidence=prob_list[pred_idx] * 100,
        probabilities=probabilities,
        probability_vector=prob_list,
    )


def group_clips_by_track(clips_dir: Path) -> dict[int, list[Path]]:
    """Group clips in `clips_dir` by track_id parsed from filename.

    Expects filenames of the form `{stem}_ID_{track_id}_clip_{NNN}.mp4`
    (as produced by `rpa.process_runners.process_video`).
    """
    grouped: dict[int, list[Path]] = {}
    if not clips_dir.exists():
        return grouped

    for clip_path in sorted(clips_dir.glob("*.mp4")):
        match = CLIP_NAME_RE.match(clip_path.name)
        if not match:
            logger.warning("Skipping unexpected clip name: {name}", name=clip_path.name)
            continue
        track_id = int(match.group("track"))
        grouped.setdefault(track_id, []).append(clip_path)

    return grouped


def select_primary_track(grouped: dict[int, list[Path]]) -> int | None:
    """Pick the track with the most clips (tie-break: lowest track_id)."""
    if not grouped:
        return None
    return max(grouped, key=lambda tid: (len(grouped[tid]), -tid))


def aggregate_predictions(
    clip_results: list[ClipResult],
    idx_to_label: dict[int, int | str],
) -> dict[str, int | str | float | dict[int | str, float]]:
    """Combine per-clip softmax vectors via mean, then argmax.

    Returns a dict with predicted_label, confidence, and mean_probabilities.
    """
    prob_matrix = np.array([r.probability_vector for r in clip_results], dtype=np.float64)
    mean_probs = prob_matrix.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    return {
        "predicted_idx": pred_idx,
        "predicted_label": idx_to_label[pred_idx],
        "confidence": float(mean_probs[pred_idx]) * 100,
        "mean_probabilities": {
            idx_to_label[i]: float(mean_probs[i]) * 100 for i in range(len(mean_probs))
        },
    }


def build_pipeline_config(args: argparse.Namespace) -> PreprocessorConfig:
    """Build PreprocessorConfig from CLI args (overriding only the exposed fields)."""
    return PreprocessorConfig(
        roi_height_ratio=args.roi_height_ratio,
        min_roi_size=args.min_roi_size,
        min_track_frames=args.min_frames,
        slice_len=args.slice_len,
        stride=args.stride,
        conf_threshold=args.conf_threshold,
        min_speed_ratio=args.min_speed_ratio,
    )


def run_end_to_end(
    model_dir: Path,
    video_path: Path,
    workdir: Path,
    pipeline_config: PreprocessorConfig,
) -> dict:
    """Run the full pipeline + per-clip inference + aggregation."""
    model, idx_to_label, device = load_model(model_dir)

    pipeline_out = workdir / "pipeline_output"
    pipeline_out.mkdir(parents=True, exist_ok=True)

    logger.info("Running preprocessing pipeline on: {v}", v=video_path)
    process_video(video_path=video_path, output_dir=pipeline_out, config=pipeline_config)

    clips_dir = pipeline_out / "clips"
    grouped = group_clips_by_track(clips_dir)
    total_runners = len(grouped)
    logger.info("Pipeline detected {n} runner track(s)", n=total_runners)

    base_report: dict = {
        "video": video_path.name,
        "model_dir": str(model_dir),
        "total_runners_detected": total_runners,
        "primary_track_id": None,
        "primary_track_selection_reason": None,
        "num_clips": 0,
        "predicted_label": None,
        "confidence": None,
        "mean_probabilities": None,
        "per_clip": [],
    }

    if total_runners == 0:
        logger.warning("No runners detected; returning empty result")
        return base_report

    primary_track = select_primary_track(grouped)
    assert primary_track is not None  # guarded by total_runners check
    primary_clips = grouped[primary_track]
    logger.info(
        "Selected primary track_id={tid} ({n} clips). Other tracks: {others}",
        tid=primary_track,
        n=len(primary_clips),
        others={tid: len(clips) for tid, clips in grouped.items() if tid != primary_track},
    )

    clip_results: list[ClipResult] = []
    for clip_path in primary_clips:
        result = predict_clip(model, clip_path, idx_to_label, device)
        if result is None:
            logger.warning("Inference failed for {name}", name=clip_path.name)
            continue
        clip_results.append(result)
        logger.info(
            "Clip {name}: {label} ({conf:.1f}%)",
            name=result.clip_name,
            label=result.predicted_label,
            conf=result.confidence,
        )

    if not clip_results:
        logger.error("All clip inferences failed for primary track {tid}", tid=primary_track)
        base_report["primary_track_id"] = primary_track
        base_report["primary_track_selection_reason"] = "most_clips"
        return base_report

    aggregated = aggregate_predictions(clip_results, idx_to_label)
    base_report.update(
        {
            "primary_track_id": primary_track,
            "primary_track_selection_reason": "most_clips",
            "num_clips": len(clip_results),
            "predicted_label": aggregated["predicted_label"],
            "confidence": aggregated["confidence"],
            "mean_probabilities": aggregated["mean_probabilities"],
            "per_clip": [
                {
                    "clip": r.clip_name,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "probabilities": r.probabilities,
                }
                for r in clip_results
            ],
        }
    )
    return base_report


def log_summary(report: dict) -> None:
    """Log a concise human-readable summary of the report."""
    logger.info("=" * 60)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 60)
    logger.info("Video:                {v}", v=report["video"])
    logger.info("Runners detected:     {n}", n=report["total_runners_detected"])
    if report["predicted_label"] is None:
        logger.info("No prediction available.")
        logger.info("=" * 60)
        return
    logger.info("Primary track_id:     {t}", t=report["primary_track_id"])
    logger.info("Clips used:           {n}", n=report["num_clips"])
    logger.info("Predicted pattern:    {label}", label=report["predicted_label"])
    logger.info("Confidence:           {c:.1f}%", c=report["confidence"])
    logger.info("Class probabilities:")
    for label, pct in report["mean_probabilities"].items():
        logger.info("  {label}: {p:.1f}%", label=label, p=pct)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end running pattern inference on a raw trackside video "
            "using a trained VideoMAE model."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local directory containing the trained VideoMAE model files "
             "(config.json, model weights, label_mapping.json).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the raw input video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the JSON report (default: print to stdout only).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for intermediate pipeline output "
             "(default: auto-created temp dir, deleted on exit).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the working directory (extracted clips, stabilized videos) after running.",
    )

    # Pipeline config knobs
    parser.add_argument("--min-frames", type=int, default=20,
                        help="Minimum frames per track to survive ghost filter (default: 20).")
    parser.add_argument("--roi-height-ratio", type=float, default=0.40,
                        help="Foot-region crop size as fraction of bbox height (default: 0.40).")
    parser.add_argument("--min-roi-size", type=int, default=128,
                        help="Minimum ROI crop size in pixels (default: 128).")
    parser.add_argument("--slice-len", type=int, default=16,
                        help="Frames per clip (default: 16, matches VideoMAE input).")
    parser.add_argument("--stride", type=int, default=16,
                        help="Stride between clips (default: 16, no overlap).")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25).")
    parser.add_argument("--min-speed-ratio", type=float, default=0.5,
                        help="Keep tracks with speed >= ratio * max_speed (default: 0.5).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_dir.exists():
        logger.error("Model directory does not exist: {p}", p=args.model_dir)
        sys.exit(1)
    if not args.video.exists():
        logger.error("Input video does not exist: {p}", p=args.video)
        sys.exit(1)

    if args.workdir is not None:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup_workdir = not args.keep_intermediates
    else:
        workdir = Path(tempfile.mkdtemp(prefix="rpa_inference_"))
        cleanup_workdir = not args.keep_intermediates
    logger.info("Working directory: {p} (cleanup={c})", p=workdir, c=cleanup_workdir)

    try:
        pipeline_config = build_pipeline_config(args)
        report = run_end_to_end(
            model_dir=args.model_dir,
            video_path=args.video,
            workdir=workdir,
            pipeline_config=pipeline_config,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Wrote JSON report to: {p}", p=args.output)
        else:
            print(json.dumps(report, indent=2, default=str))
    finally:
        if cleanup_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Cleaned up workdir: {p}", p=workdir)

    # Print summary LAST so it stays on screen after all other output.
    log_summary(report)


if __name__ == "__main__":
    main()
