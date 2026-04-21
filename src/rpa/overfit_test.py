"""Overfit test script for VideoMAE training pipeline verification.

This script intentionally overfits a VideoMAE model on a small dataset to verify
the training pipeline is working correctly. Success criteria: 100% accuracy on
the same data used for training (memorization test).

Usage:
    uv run python -m rpa.overfit_test --data_dir /path/to/clips

The script expects video files with naming pattern: *_label_N_*.mp4
where N is the class label integer.
"""

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE requirements
NUM_FRAMES = 16
FRAME_SIZE = 224


class OverfitDataset(Dataset):
    """Dataset for loading video clips with labels extracted from filenames.

    Filename Pattern: *_label_N_*.mp4 where N is the class label.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize dataset by scanning directory for video files.

        Args:
            data_dir: Path to directory containing .mp4 video files
        """
        self.data_dir = Path(data_dir)
        self.video_files: list[Path] = []
        self.labels: list[int] = []
        self.label_pattern = re.compile(r"_label_(\d+)_")

        # Scan for video files and extract labels
        self._scan_videos()

        # Build label mapping (original label -> contiguous index)
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        logger.info(
            "Loaded {n} videos with {c} classes: {labels}",
            n=len(self.video_files),
            c=self.num_classes,
            labels=unique_labels,
        )

    def _scan_videos(self) -> None:
        """Scan directory for .mp4 files and extract labels."""
        for video_path in sorted(self.data_dir.glob("*.mp4")):
            match = self.label_pattern.search(video_path.name)
            if match:
                label = int(match.group(1))
                self.video_files.append(video_path)
                self.labels.append(label)
            else:
                logger.warning(
                    "Skipping {name}: no label pattern found",
                    name=video_path.name,
                )

        if not self.video_files:
            msg = f"No valid video files found in {self.data_dir}"
            raise ValueError(msg)

    def _load_video_frames(self, video_path: Path) -> np.ndarray | None:
        """Load and preprocess video frames.

        Args:
            video_path: Path to video file

        Returns:
            Array of shape (NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3) normalized,
            or None if loading fails
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video: {path}", path=video_path)
            return None

        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR
                )
                frames.append(frame_resized)
        finally:
            cap.release()

        if not frames:
            logger.warning("No frames extracted from: {path}", path=video_path)
            return None

        # Temporal sampling: ensure exactly NUM_FRAMES
        frames_array = self._temporal_sample(frames)

        # Normalize to [0, 1] then apply ImageNet normalization
        frames_normalized = frames_array.astype(np.float32) / 255.0
        frames_normalized = (frames_normalized - IMAGENET_MEAN) / IMAGENET_STD

        result: np.ndarray = frames_normalized.astype(np.float32)
        return result

    def _temporal_sample(self, frames: list[np.ndarray]) -> np.ndarray:
        """Sample exactly NUM_FRAMES from the video.

        Strategy:
        - If video > NUM_FRAMES: take center NUM_FRAMES
        - If video < NUM_FRAMES: loop/repeat frames to reach NUM_FRAMES

        Args:
            frames: List of video frames

        Returns:
            Array of shape (NUM_FRAMES, H, W, 3)
        """
        n_frames = len(frames)

        if n_frames == NUM_FRAMES:
            return np.stack(frames)

        if n_frames > NUM_FRAMES:
            # Take center frames
            start = (n_frames - NUM_FRAMES) // 2
            return np.stack(frames[start : start + NUM_FRAMES])

        # n_frames < NUM_FRAMES: loop frames
        indices = []
        for i in range(NUM_FRAMES):
            indices.append(i % n_frames)
        return np.stack([frames[i] for i in indices])

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """Get a video sample.

        Returns:
            Dictionary with:
            - pixel_values: Tensor of shape (3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
            - label: Class index (int)
            - filename: Video filename for debugging
        """
        video_path = self.video_files[idx]
        original_label = self.labels[idx]
        label_idx = self.label_to_idx[original_label]

        try:
            frames = self._load_video_frames(video_path)
            if frames is None:
                # Return a zero tensor as fallback for corrupt videos
                frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
        except Exception as e:
            logger.warning("Error loading {path}: {err}", path=video_path, err=e)
            frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        # Convert from (T, H, W, C) to (T, C, H, W) for VideoMAE
        # VideoMAE expects shape: (batch, num_frames, num_channels, height, width)
        frames_tensor: torch.Tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return {
            "pixel_values": frames_tensor,
            "label": label_idx,
            "filename": video_path.name,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate function to batch video samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with stacked tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    filenames = [item["filename"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "filenames": filenames,
    }


def train_overfit(
    data_dir: Path,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
) -> tuple[VideoMAEForVideoClassification, OverfitDataset]:
    """Train VideoMAE model to overfit on the dataset.

    Args:
        data_dir: Path to video clips directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for AdamW optimizer

    Returns:
        Tuple of (trained model, dataset)
    """
    # Device setup: prefer CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {device}", device=device)

    # Load dataset
    dataset = OverfitDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with cv2
        collate_fn=collate_fn,
    )

    # Load model with dynamic number of classes
    logger.info("Loading VideoMAE model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=dataset.num_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting overfit training for {n} epochs...", n=epochs)
    total_steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)
            total_steps += 1

            logger.info(
                "Step {step} | Loss: {loss:.4f}",
                step=total_steps,
                loss=loss.item(),
            )

        # Epoch summary
        epoch_acc = epoch_correct / epoch_total * 100
        avg_loss = epoch_loss / len(dataloader)
        logger.info(
            "Epoch {epoch}/{total} | Avg Loss: {loss:.4f} | Accuracy: {acc:.1f}%",
            epoch=epoch + 1,
            total=epochs,
            loss=avg_loss,
            acc=epoch_acc,
        )

    return model, dataset


def verify_predictions(
    model: VideoMAEForVideoClassification,
    dataset: OverfitDataset,
    num_samples: int = 5,
) -> None:
    """Verify model predictions on random samples from the dataset.

    Args:
        model: Trained VideoMAE model
        dataset: Dataset to sample from
        num_samples: Number of samples to verify
    """
    device = next(model.parameters()).device
    model.eval()

    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    logger.info("=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)

    # Build results table
    header = f"{'Filename':<45} | {'Pred':>5} | {'Actual':>6} | {'Conf':>7}"
    separator = "-" * len(header)
    results_lines = [separator, header, separator]

    correct = 0
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            # Type assertions for mypy
            pixel_values_tensor = sample["pixel_values"]
            assert isinstance(pixel_values_tensor, torch.Tensor)
            pixel_values = pixel_values_tensor.unsqueeze(0).to(device)

            actual_idx = sample["label"]
            assert isinstance(actual_idx, int)

            filename = sample["filename"]
            assert isinstance(filename, str)

            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = int(outputs.logits.argmax(dim=-1).item())
            confidence = float(probs[0, pred_idx].item()) * 100

            # Convert back to original labels
            pred_label = dataset.idx_to_label[pred_idx]
            actual_label = dataset.idx_to_label[actual_idx]

            is_correct = pred_idx == actual_idx
            if is_correct:
                correct += 1

            status = "OK" if is_correct else "FAIL"
            results_lines.append(
                f"{filename:<45} | {pred_label:>5} | {actual_label:>6} | {confidence:>6.1f}% {status}"
            )

    results_lines.append(separator)
    results_lines.append(
        f"Verification Accuracy: {correct}/{len(indices)} ({correct / len(indices) * 100:.0f}%)"
    )
    results_lines.append(separator)

    # Log all results
    for line in results_lines:
        logger.info(line)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Overfit VideoMAE on a small dataset to verify training pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to directory containing video clips",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--verify_samples",
        type=int,
        default=5,
        help="Number of samples to verify after training (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate input directory
    if not args.data_dir.exists():
        logger.error("Data directory does not exist: {path}", path=args.data_dir)
        sys.exit(1)

    if not args.data_dir.is_dir():
        logger.error("Path is not a directory: {path}", path=args.data_dir)
        sys.exit(1)

    logger.info("Starting overfit test with data from: {path}", path=args.data_dir)

    # Train model
    model, dataset = train_overfit(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Verify predictions
    verify_predictions(model, dataset, num_samples=args.verify_samples)

    logger.success("Overfit test complete!")

    # Save model to disk
    save_dir = args.data_dir.parent / "trained_model"
    save_model(model, dataset, save_dir)

    # Run prediction on test video
    test_video = Path(__file__).parent.parent.parent / "output_test/clips/output_ID_2_clip_003.mp4"
    predict_single_video(model, dataset, test_video)


def save_model(
    model: VideoMAEForVideoClassification,
    dataset: OverfitDataset,
    save_dir: Path,
) -> None:
    """Save trained model and label mapping to disk.

    Args:
        model: Trained VideoMAE model
        dataset: Dataset (for label mapping)
        save_dir: Directory to save model files
    """
    import json  # noqa: PLC0415

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the model using Hugging Face's save_pretrained
    model.save_pretrained(save_dir)
    logger.info("Model saved to: {path}", path=save_dir)

    # Save label mapping
    label_mapping = {
        "idx_to_label": dataset.idx_to_label,
        "label_to_idx": dataset.label_to_idx,
        "num_classes": dataset.num_classes,
    }
    label_path = save_dir / "label_mapping.json"
    with label_path.open("w") as f:
        json.dump(label_mapping, f, indent=2)
    logger.info("Label mapping saved to: {path}", path=label_path)


def load_video_for_inference(video_path: Path) -> torch.Tensor | None:
    """Load and preprocess a video file for inference.

    Args:
        video_path: Path to video file

    Returns:
        Tensor of shape (1, T, C, H, W) ready for model, or None if loading fails
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
            frames.append(frame_resized)
    finally:
        cap.release()

    if not frames:
        logger.error("No frames extracted from video")
        return None

    # Temporal sampling
    n_frames = len(frames)
    if n_frames == NUM_FRAMES:
        frames_array = np.stack(frames)
    elif n_frames > NUM_FRAMES:
        start = (n_frames - NUM_FRAMES) // 2
        frames_array = np.stack(frames[start : start + NUM_FRAMES])
    else:
        indices = [i % n_frames for i in range(NUM_FRAMES)]
        frames_array = np.stack([frames[i] for i in indices])

    # Normalize
    frames_normalized = frames_array.astype(np.float32) / 255.0
    frames_normalized = (frames_normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to tensor: (T, H, W, C) -> (1, T, C, H, W)
    frames_tensor = torch.from_numpy(frames_normalized.astype(np.float32)).permute(0, 3, 1, 2)
    return frames_tensor.unsqueeze(0)


def predict_single_video(
    model: VideoMAEForVideoClassification,
    dataset: OverfitDataset,
    video_path: Path,
) -> None:
    """Run prediction on a single video file.

    Args:
        model: Trained VideoMAE model
        dataset: Dataset (used for label mapping)
        video_path: Path to video file
    """
    if not video_path.exists():
        logger.warning("Test video not found: {path}", path=video_path)
        return

    device = next(model.parameters()).device
    model.eval()

    logger.info("=" * 60)
    logger.info("SINGLE VIDEO PREDICTION")
    logger.info("=" * 60)
    logger.info("Video: {path}", path=video_path)

    pixel_values = load_video_for_inference(video_path)
    if pixel_values is None:
        return

    pixel_values = pixel_values.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = int(outputs.logits.argmax(dim=-1).item())
        confidence = float(probs[0, pred_idx].item()) * 100

    pred_label = dataset.idx_to_label[pred_idx]
    logger.info("Predicted Label: {label}", label=pred_label)
    logger.info("Confidence: {conf:.1f}%", conf=confidence)
    logger.info("All class probabilities:")
    for idx, prob in enumerate(probs[0].tolist()):
        label = dataset.idx_to_label[idx]
        logger.info("  Class {label}: {prob:.1f}%", label=label, prob=prob * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
