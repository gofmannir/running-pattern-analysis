"""Offline video augmentation script for generating augmented dataset.

This script generates multiple augmented versions of each video in the dataset,
applying random combinations of augmentations. Supports local files and GCS buckets.

Usage:
    # Local directory
    uv run python -m rpa.augment \
        --input /path/to/videos/ \
        --output /path/to/augmented/ \
        --versions 25

    # GCS bucket
    uv run python -m rpa.augment \
        --input gs://bucket/raw/ \
        --output gs://bucket/augmented/ \
        --versions 25 \
        --workers 4

    # Single video (for testing)
    uv run python -m rpa.augment \
        --input /path/to/video.mp4 \
        --output /path/to/output/ \
        --versions 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass
class AugmentationParams:
    """Parameters for a single augmentation run."""

    do_flip: bool = False
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    do_blur: bool = False
    blur_sigma: float = 0.0
    do_rotation: bool = False
    rotation_angle: float = 0.0
    do_scale: bool = False
    scale_factor: float = 1.0
    do_cutout: bool = False
    cutout_top: float = 0.0  # As fraction of height
    cutout_left: float = 0.0  # As fraction of width
    cutout_ratio: float = 0.15

    def describe(self) -> str:
        """Return human-readable description of augmentations applied."""
        parts = ["gray"]
        if self.do_flip:
            parts.append("flip")
        if self.brightness_delta != 0 or self.contrast_delta != 0:
            parts.append(f"bc({self.brightness_delta:.2f},{self.contrast_delta:.2f})")
        if self.do_blur:
            parts.append(f"blur({self.blur_sigma:.2f})")
        if self.do_rotation:
            parts.append(f"rot({self.rotation_angle:.1f})")
        if self.do_scale:
            parts.append(f"scale({self.scale_factor:.2f})")
        if self.do_cutout:
            parts.append("cutout")
        return "+".join(parts)


@dataclass
class AugmentConfig:
    """Configuration for augmentation generation."""

    versions_per_video: int = 25
    # Probabilities for each augmentation type
    flip_prob: float = 0.7
    brightness_contrast_prob: float = 0.8
    blur_prob: float = 0.6
    rotation_prob: float = 0.8
    scale_prob: float = 0.6
    cutout_prob: float = 0.6
    # Augmentation parameters
    brightness_range: tuple[float, float] = field(default_factory=lambda: (-0.25, 0.25))
    contrast_range: tuple[float, float] = field(default_factory=lambda: (-0.25, 0.25))
    blur_sigma_range: tuple[float, float] = field(default_factory=lambda: (0.1, 1.0))
    rotation_range: tuple[float, float] = field(default_factory=lambda: (-5.0, 5.0))
    scale_range: tuple[float, float] = field(default_factory=lambda: (0.9, 1.1))
    cutout_ratio: float = 0.15


def create_selective_config(
    enabled_augmentations: list[str],
    versions_per_video: int = 25,
) -> AugmentConfig:
    """Create an AugmentConfig with only specified augmentations enabled.

    This is useful for systematic experiments where you want to test the effect
    of individual augmentations or specific combinations.

    Args:
        enabled_augmentations: List of augmentation names to enable.
            Valid names: "flip", "brightness_contrast", "blur",
                        "rotation", "scale", "cutout"
        versions_per_video: Number of augmented versions per video.

    Returns:
        AugmentConfig with only specified augmentations enabled at standard probabilities.
        All other augmentations will have probability 0.0.

    Example:
        >>> config = create_selective_config(["flip", "blur"], versions_per_video=10)
        >>> config.flip_prob
        0.7
        >>> config.blur_prob
        0.6
        >>> config.rotation_prob
        0.0
    """
    # Map of augmentation names to their (param_name, standard_prob)
    aug_map: dict[str, tuple[str, float]] = {
        "flip": ("flip_prob", 0.7),
        "brightness_contrast": ("brightness_contrast_prob", 0.8),
        "blur": ("blur_prob", 0.6),
        "rotation": ("rotation_prob", 0.8),
        "scale": ("scale_prob", 0.6),
        "cutout": ("cutout_prob", 0.6),
    }

    # Start with all augmentations disabled
    config = AugmentConfig(
        versions_per_video=versions_per_video,
        flip_prob=0.0,
        brightness_contrast_prob=0.0,
        blur_prob=0.0,
        rotation_prob=0.0,
        scale_prob=0.0,
        cutout_prob=0.0,
    )

    # Enable only the specified augmentations
    for aug_name in enabled_augmentations:
        aug_name_lower = aug_name.lower()
        if aug_name_lower in aug_map:
            param_name, prob = aug_map[aug_name_lower]
            setattr(config, param_name, prob)
        else:
            logger.warning(
                "Unknown augmentation: {name}. Valid names: {valid}",
                name=aug_name,
                valid=list(aug_map.keys()),
            )

    return config


def generate_augmentation_params(
    video_path: str, version: int, config: AugmentConfig
) -> AugmentationParams:
    """Generate deterministic random augmentation parameters for a video version.

    Uses hash of video path + version number as seed for reproducibility.
    """
    # Create deterministic seed from video path and version
    seed_str = f"{video_path}_{version}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    params = AugmentationParams()

    # Decide which augmentations to apply
    params.do_flip = rng.random() < config.flip_prob

    if rng.random() < config.brightness_contrast_prob:
        params.brightness_delta = rng.uniform(*config.brightness_range)
        params.contrast_delta = rng.uniform(*config.contrast_range)

    if rng.random() < config.blur_prob:
        params.do_blur = True
        params.blur_sigma = rng.uniform(*config.blur_sigma_range)

    if rng.random() < config.rotation_prob:
        params.do_rotation = True
        params.rotation_angle = rng.uniform(*config.rotation_range)

    if rng.random() < config.scale_prob:
        params.do_scale = True
        params.scale_factor = rng.uniform(*config.scale_range)

    if rng.random() < config.cutout_prob:
        params.do_cutout = True
        # Restrict cutout to upper 50% of frame to avoid hiding runner's feet
        params.cutout_top = rng.uniform(0, 0.5 - config.cutout_ratio)
        params.cutout_left = rng.uniform(0, 1 - config.cutout_ratio)
        params.cutout_ratio = config.cutout_ratio

    return params


# ============================================================================
# Frame-level augmentation functions
# ============================================================================


def to_grayscale(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Convert RGB frames to grayscale (replicated to 3 channels)."""
    grayscale_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray, gray, gray], axis=-1)
        grayscale_frames.append(gray_3ch)
    return grayscale_frames


def apply_horizontal_flip(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply horizontal flip to all frames."""
    return [cv2.flip(frame, 1) for frame in frames]


def apply_brightness_contrast(
    frames: list[np.ndarray], brightness_delta: float, contrast_delta: float
) -> list[np.ndarray]:
    """Apply brightness and contrast adjustment to all frames."""
    brightness = 1.0 + brightness_delta
    contrast = 1.0 + contrast_delta

    adjusted_frames = []
    for frame in frames:
        frame_float = frame.astype(np.float32)
        frame_adjusted = contrast * (frame_float - 128.0) + 128.0 + (brightness - 1.0) * 255.0
        frame_adjusted = np.clip(frame_adjusted, 0, 255).astype(np.uint8)
        adjusted_frames.append(frame_adjusted)

    return adjusted_frames


def apply_gaussian_blur(frames: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    """Apply Gaussian blur to all frames."""
    min_kernel = 3
    kernel_size = max(int(2 * round(3 * sigma) + 1), min_kernel)
    return [cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma) for frame in frames]


def apply_rotation(frames: list[np.ndarray], angle: float) -> list[np.ndarray]:
    """Apply rotation to all frames."""
    h, w = frames[0].shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_frames = []
    for frame in frames:
        rotated = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        rotated_frames.append(rotated)

    return rotated_frames


def apply_scale_crop(frames: list[np.ndarray], scale: float, rng: random.Random) -> list[np.ndarray]:
    """Apply random scale and crop to all frames."""
    h, w = frames[0].shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Calculate crop position (deterministic within this call)
    if new_h > h:
        top = rng.randint(0, new_h - h)
        left = rng.randint(0, new_w - w)
    else:
        top, left = 0, 0

    scaled_frames = []
    for frame in frames:
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if new_h >= h and new_w >= w:
            cropped = scaled[top : top + h, left : left + w]
        else:
            cropped = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        scaled_frames.append(cropped)

    return scaled_frames


def apply_cutout(
    frames: list[np.ndarray], top_frac: float, left_frac: float, ratio: float
) -> list[np.ndarray]:
    """Apply cutout (random erasing) to all frames at the same location."""
    h, w = frames[0].shape[:2]
    cutout_h = int(h * np.sqrt(ratio))
    cutout_w = int(w * np.sqrt(ratio))
    top = int(top_frac * (h - cutout_h))
    left = int(left_frac * (w - cutout_w))

    cutout_frames = []
    for frame in frames:
        frame_copy = frame.copy()
        frame_copy[top : top + cutout_h, left : left + cutout_w] = 128
        cutout_frames.append(frame_copy)

    return cutout_frames


def apply_augmentations(frames: list[np.ndarray], params: AugmentationParams) -> list[np.ndarray]:
    """Apply all augmentations specified by params to frames."""
    # Always apply grayscale
    frames = to_grayscale(frames)

    if params.do_flip:
        frames = apply_horizontal_flip(frames)

    if params.brightness_delta != 0 or params.contrast_delta != 0:
        frames = apply_brightness_contrast(frames, params.brightness_delta, params.contrast_delta)

    if params.do_blur:
        frames = apply_gaussian_blur(frames, params.blur_sigma)

    if params.do_rotation:
        frames = apply_rotation(frames, params.rotation_angle)

    if params.do_scale:
        # Use deterministic RNG for scale crop position
        seed_str = f"{params.scale_factor}_{params.rotation_angle}"
        rng = random.Random(hash(seed_str))
        frames = apply_scale_crop(frames, params.scale_factor, rng)

    if params.do_cutout:
        frames = apply_cutout(frames, params.cutout_top, params.cutout_left, params.cutout_ratio)

    return frames


# ============================================================================
# Video I/O functions
# ============================================================================


def read_video_frames(video_path: Path) -> list[np.ndarray] | None:
    """Read all frames from a video file."""
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
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        logger.error("No frames extracted from video: {path}", path=video_path)
        return None

    return frames


def write_video_frames(
    frames: list[np.ndarray], output_path: Path, fps: float = 30.0
) -> bool:
    """Write frames to a video file using lossless H.264 encoding via ffmpeg."""
    if not frames:
        return False

    h, w = frames[0].shape[:2]

    # Use ffmpeg for lossless H.264 encoding (CRF 0 = lossless)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",  # Read from stdin
        "-c:v",
        "libx264",
        "-crf",
        "0",  # Lossless
        "-preset",
        "ultrafast",  # Fast encoding, still lossless at CRF 0
        "-pix_fmt",
        "yuv444p",  # Preserve full color info for lossless
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames:
            # Write RGB frame directly to ffmpeg
            proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait()
        success = proc.returncode == 0
    except Exception as e:
        logger.error("Failed to write video with ffmpeg: {e}", e=e)
        success = False

    return success


def get_video_fps(video_path: Path) -> float:
    """Get FPS of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


# ============================================================================
# GCS utilities
# ============================================================================


def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS path."""
    return path.startswith("gs://")


def gcs_list_files(gcs_path: str, extension: str = ".mp4") -> list[str]:
    """List files in a GCS path."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True,
        text=True,
        check=True,
    )
    files = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    return [f for f in files if f.endswith(extension)]


def gcs_download(gcs_path: str, local_path: Path) -> bool:
    """Download a file from GCS."""
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, str(local_path)],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to download {gcs}: {err}", gcs=gcs_path, err=e.stderr)
        return False
    return True


def gcs_upload(local_path: Path, gcs_path: str) -> bool:
    """Upload a file to GCS."""
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_path],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to upload to {gcs}: {err}", gcs=gcs_path, err=e.stderr)
        return False
    return True


def gcs_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists."""
    try:
        subprocess.run(
            ["gcloud", "storage", "ls", gcs_path],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def gcs_read_json(gcs_path: str) -> dict | None:
    """Read a JSON file from GCS."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if gcs_download(gcs_path, tmp_path):
            with tmp_path.open() as f:
                return json.load(f)
    finally:
        tmp_path.unlink(missing_ok=True)
    return None


def gcs_write_json(data: dict, gcs_path: str) -> bool:
    """Write a JSON file to GCS."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)
    try:
        return gcs_upload(tmp_path, gcs_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ============================================================================
# Checkpoint management
# ============================================================================


@dataclass
class Checkpoint:
    """Checkpoint for resuming augmentation process."""

    completed: set[str] = field(default_factory=set)
    total: int = 0
    started_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "completed": list(self.completed),
            "total": self.total,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        """Create from dictionary."""
        return cls(
            completed=set(data.get("completed", [])),
            total=data.get("total", 0),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
        )


def load_checkpoint(checkpoint_path: str) -> Checkpoint:
    """Load checkpoint from local file or GCS."""
    if is_gcs_path(checkpoint_path):
        data = gcs_read_json(checkpoint_path)
        if data:
            logger.info("Loaded checkpoint from GCS: {n} completed", n=len(data.get("completed", [])))
            return Checkpoint.from_dict(data)
    else:
        path = Path(checkpoint_path)
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            logger.info("Loaded checkpoint: {n} completed", n=len(data.get("completed", [])))
            return Checkpoint.from_dict(data)
    return Checkpoint()


def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: str) -> None:
    """Save checkpoint to local file or GCS."""
    checkpoint.last_updated = datetime.now(UTC).isoformat()

    if is_gcs_path(checkpoint_path):
        gcs_write_json(checkpoint.to_dict(), checkpoint_path)
    else:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)


# ============================================================================
# Core augmentation logic
# ============================================================================


def augment_single_video(
    input_path: Path | str,
    output_dir: Path | str,
    config: AugmentConfig,
    video_id: str | None = None,
) -> list[str]:
    """Augment a single video and save all versions.

    Args:
        input_path: Path to input video (local or will be downloaded)
        output_dir: Directory to save augmented videos
        config: Augmentation configuration
        video_id: Identifier for the video (used for deterministic seeds)

    Returns:
        List of output file paths
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_id is None:
        video_id = input_path.stem

    # Read video
    frames = read_video_frames(input_path)
    if frames is None:
        return []

    fps = get_video_fps(input_path)
    output_paths: list[str] = []

    # Generate augmented versions
    for version in range(1, config.versions_per_video + 1):
        # Generate augmentation parameters
        params = generate_augmentation_params(video_id, version, config)

        # Apply augmentations
        augmented_frames = apply_augmentations(frames.copy(), params)

        # Write output - insert version before label (last segment after _)
        parts = input_path.stem.rsplit("_", 1)
        if len(parts) == 2:
            output_name = f"{parts[0]}_v{version:03d}_{parts[1]}.mp4"
        else:
            output_name = f"{input_path.stem}_v{version:03d}.mp4"
        output_path = output_dir / output_name

        if write_video_frames(augmented_frames, output_path, fps):
            output_paths.append(str(output_path))
            logger.debug(
                "Created {path} ({desc})",
                path=output_name,
                desc=params.describe(),
            )

    return output_paths


def process_video_worker(args: tuple) -> tuple[str, list[str], bool]:
    """Worker function for parallel processing.

    Args:
        args: Tuple of (video_path, output_dir, config, is_gcs_input, is_gcs_output, gcs_output_prefix)

    Returns:
        Tuple of (video_id, output_paths, success)
    """
    video_path, output_dir, config, is_gcs_input, is_gcs_output, gcs_output_prefix = args

    video_id = Path(video_path).stem if not is_gcs_path(video_path) else video_path.split("/")[-1].replace(".mp4", "")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Download if GCS
            if is_gcs_input:
                local_input = tmp_path / f"{video_id}.mp4"
                if not gcs_download(video_path, local_input):
                    return (video_id, [], False)
            else:
                local_input = Path(video_path)

            # Set output directory
            local_output_dir = tmp_path / "output" if is_gcs_output else Path(output_dir)

            # Run augmentation
            output_paths = augment_single_video(local_input, local_output_dir, config, video_id)

            if not output_paths:
                return (video_id, [], False)

            # Upload if GCS output
            if is_gcs_output:
                gcs_paths = []
                for local_path in output_paths:
                    gcs_path = f"{gcs_output_prefix}{Path(local_path).name}"
                    if gcs_upload(Path(local_path), gcs_path):
                        gcs_paths.append(gcs_path)
                return (video_id, gcs_paths, True)

            return (video_id, output_paths, True)

    except Exception as e:
        logger.error("Error processing {vid}: {err}", vid=video_id, err=e)
        return (video_id, [], False)


def _collect_video_paths(input_path: str) -> list[str]:
    """Collect video paths from input (file, directory, or GCS)."""
    if is_gcs_path(input_path):
        logger.info("Listing videos from GCS: {path}", path=input_path)
        video_paths = gcs_list_files(input_path)
        logger.info("Found {n} videos in GCS", n=len(video_paths))
    elif Path(input_path).is_file():
        video_paths = [input_path]
    else:
        input_dir = Path(input_path)
        video_paths = [str(p) for p in sorted(input_dir.glob("*.mp4"))]
        logger.info("Found {n} videos in directory", n=len(video_paths))
    return video_paths


def run_augmentation(
    input_path: str,
    output_path: str,
    config: AugmentConfig,
    workers: int = 1,
    checkpoint_path: str | None = None,
    video_paths: list[str] | None = None,
) -> dict[str, list[str]]:
    """Run augmentation on input (file, directory, or GCS path).

    Args:
        input_path: Input path (file, directory, or gs://...). Used to determine
            GCS input mode. Ignored if video_paths is provided.
        output_path: Output path (directory or gs://...)
        config: Augmentation configuration
        workers: Number of parallel workers
        checkpoint_path: Path to checkpoint file (local or GCS)
        video_paths: Optional list of specific video paths to process.
            If provided, input_path is only used to determine GCS mode.

    Returns:
        Dictionary mapping original video IDs to list of augmented paths
    """
    is_gcs_input = is_gcs_path(input_path)
    is_gcs_output = is_gcs_path(output_path)

    # Collect input videos (use provided list or collect from input_path)
    if video_paths is not None:
        collected_paths = video_paths
        # Update GCS input flag based on actual paths
        if collected_paths:
            is_gcs_input = is_gcs_path(collected_paths[0])
    else:
        collected_paths = _collect_video_paths(input_path)

    if not collected_paths:
        logger.error("No videos found in {path}", path=input_path)
        return {}

    # Load checkpoint
    checkpoint = Checkpoint()
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        if not checkpoint.started_at:
            checkpoint.started_at = datetime.now(UTC).isoformat()
        checkpoint.total = len(collected_paths)

    # Filter already completed
    pending_paths = [p for p in collected_paths if p not in checkpoint.completed]
    logger.info(
        "Processing {pending}/{total} videos ({done} already done)",
        pending=len(pending_paths),
        total=len(collected_paths),
        done=len(checkpoint.completed),
    )

    if not pending_paths:
        logger.info("All videos already processed!")
        return {}

    # Prepare output prefix for GCS
    gcs_output_prefix = output_path.rstrip("/") + "/" if is_gcs_output else ""

    # Create output directory if local
    if not is_gcs_output:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Prepare worker arguments
    worker_args = [
        (path, output_path, config, is_gcs_input, is_gcs_output, gcs_output_prefix)
        for path in pending_paths
    ]

    results: dict[str, list[str]] = {}
    completed_count = len(checkpoint.completed)
    total_count = len(collected_paths)

    # Process with parallel workers
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_worker, args): args[0] for args in worker_args}

        for future in as_completed(futures):
            video_path = futures[future]
            try:
                video_id, output_paths, success = future.result()

                if success:
                    results[video_id] = output_paths
                    checkpoint.completed.add(video_path)
                    completed_count += 1

                    # Save checkpoint periodically (every 10 videos)
                    if checkpoint_path and completed_count % 10 == 0:
                        save_checkpoint(checkpoint, checkpoint_path)

                    logger.info(
                        "[{done}/{total}] {vid}: {n} versions",
                        done=completed_count,
                        total=total_count,
                        vid=video_id,
                        n=len(output_paths),
                    )
                else:
                    logger.warning("Failed to process: {vid}", vid=video_id)

            except Exception as e:
                logger.error("Error in future for {path}: {err}", path=video_path, err=e)

    # Final checkpoint save
    if checkpoint_path:
        save_checkpoint(checkpoint, checkpoint_path)

    logger.info("Augmentation complete: {n} videos processed", n=len(results))
    return results


# ============================================================================
# Dataset JSON generation
# ============================================================================


def generate_augmented_dataset_json(
    original_split_json: Path,
    augmented_prefix: str,
    output_json: Path,
    versions: int,
) -> None:
    """Generate dataset split JSON for augmented dataset.

    Args:
        original_split_json: Path to original dataset_split.json
        augmented_prefix: Prefix where augmented videos are stored
        output_json: Path to save new JSON
        versions: Number of versions per video
    """
    with original_split_json.open() as f:
        original = json.load(f)

    augmented_prefix = augmented_prefix.rstrip("/") + "/"

    new_split = {"train": [], "val": [], "test": []}

    # For training: include all augmented versions
    for path in original["train"]:
        video_stem = Path(path).stem
        # Insert version before label (last segment after _)
        parts = video_stem.rsplit("_", 1)
        for v in range(1, versions + 1):
            if len(parts) == 2:
                aug_path = f"{augmented_prefix}{parts[0]}_v{v:03d}_{parts[1]}.mp4"
            else:
                aug_path = f"{augmented_prefix}{video_stem}_v{v:03d}.mp4"
            new_split["train"].append(aug_path)

    # For val/test: only include grayscale version (v001 has minimal augmentation typically,
    # but we should generate a special "val" version with only grayscale)
    # For simplicity, use v001 which always has grayscale
    for split in ["val", "test"]:
        for path in original[split]:
            video_stem = Path(path).stem
            # Use original video path - val/test don't need augmentation
            # They should be processed separately with only grayscale
            new_split[split].append(path)

    # Save
    with output_json.open("w") as f:
        json.dump(new_split, f, indent=2)

    logger.info(
        "Generated {path}: train={t}, val={v}, test={te}",
        path=output_json,
        t=len(new_split["train"]),
        v=len(new_split["val"]),
        te=len(new_split["test"]),
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate augmented video dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path: video file, directory, or gs://bucket/prefix/",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path: directory or gs://bucket/prefix/",
    )
    parser.add_argument(
        "--versions",
        type=int,
        default=25,
        help="Number of augmented versions per video (default: 25)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file path (local or gs://...) for resume support",
    )
    parser.add_argument(
        "--generate-json",
        type=str,
        metavar="ORIGINAL_JSON",
        help="Generate augmented dataset JSON from original split JSON",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="dataset_split_augmented.json",
        help="Output path for generated dataset JSON",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create config
    config = AugmentConfig(versions_per_video=args.versions)

    logger.info("=" * 60)
    logger.info("OFFLINE VIDEO AUGMENTATION")
    logger.info("=" * 60)
    logger.info("Input: {path}", path=args.input)
    logger.info("Output: {path}", path=args.output)
    logger.info("Versions per video: {n}", n=args.versions)
    logger.info("Workers: {n}", n=args.workers)
    if args.checkpoint:
        logger.info("Checkpoint: {path}", path=args.checkpoint)
    logger.info("=" * 60)

    # Run augmentation
    results = run_augmentation(
        input_path=args.input,
        output_path=args.output,
        config=config,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
    )

    # Generate dataset JSON if requested
    if args.generate_json:
        generate_augmented_dataset_json(
            original_split_json=Path(args.generate_json),
            augmented_prefix=args.output,
            output_json=Path(args.json_output),
            versions=args.versions,
        )

    logger.info("Done! Processed {n} videos", n=len(results))


if __name__ == "__main__":
    main()
