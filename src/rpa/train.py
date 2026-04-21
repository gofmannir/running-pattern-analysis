"""Training script for VideoMAE fine-tuning on running pattern classification.

Loads dataset splits from JSON file and trains a VideoMAE model for binary classification.
Includes comprehensive augmentations to prevent overfitting and focus on foot patterns.

Usage:
    uv run python -m rpa.train --split-json dataset_split.json --output-dir trained_model

    # With custom hyperparameters
    uv run python -m rpa.train --split-json dataset_split.json --output-dir trained_model \
        --epochs 10 --batch-size 8 --lr 5e-5

    # Disable augmentations (for ablation study)
    uv run python -m rpa.train --split-json dataset_split.json --output-dir trained_model \
        --no-augmentation
"""

import argparse
import contextlib
import json
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from google.cloud import storage  # type: ignore[import-untyped]
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE requirements
NUM_FRAMES = 16
FRAME_SIZE = 224

# Training defaults
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 5e-5
LOG_INTERVAL = 10

# GCS constants
MIN_GCS_URI_PARTS = 2  # bucket + path


class GCSClientSingleton:
    """Singleton wrapper for GCS client to avoid global statement."""

    _instance: storage.Client | None = None

    @classmethod
    def get(cls) -> storage.Client:
        """Get or create a GCS client."""
        if cls._instance is None:
            cls._instance = storage.Client()
        return cls._instance


def download_gcs_to_temp(gcs_uri: str) -> Path | None:
    """Download a GCS file to a temporary local file.

    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/file.mp4)

    Returns:
        Path to temporary file, or None if download failed
    """
    if not gcs_uri.startswith("gs://"):
        return None

    # Parse URI: gs://bucket/path/to/file.mp4
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) < MIN_GCS_URI_PARTS:
        logger.warning("Invalid GCS URI: {uri}", uri=gcs_uri)
        return None

    bucket_name = parts[0]
    blob_path = parts[1]

    try:
        client = GCSClientSingleton.get()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to temp file
        suffix = Path(blob_path).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)

        blob.download_to_filename(str(tmp_path))
    except Exception as e:
        logger.warning("Failed to download {uri}: {err}", uri=gcs_uri, err=e)
        return None

    return tmp_path


@dataclass
class AugmentationConfig:
    """Configuration for video augmentations.

    All augmentations except grayscale are applied only during training.
    Grayscale is always applied to prevent the model from learning color patterns.
    """

    # Always applied (train + val + test)
    grayscale: bool = True

    # Train-only augmentations
    horizontal_flip_prob: float = 0.5
    temporal_offset: bool = True  # Random start offset for temporal sampling

    # Brightness and contrast (train only)
    brightness_delta: float = 0.25  # ±25%
    contrast_delta: float = 0.25  # ±25%

    # Scale and crop (train only)
    scale_range: tuple[float, float] = field(default_factory=lambda: (0.9, 1.1))

    # Gaussian blur (train only)
    blur_prob: float = 0.3
    blur_sigma_max: float = 1.0

    # Small rotation (train only)
    rotation_prob: float = 0.3
    rotation_degrees: float = 5.0

    # Cutout / random erasing (train only)
    cutout_prob: float = 0.3
    cutout_ratio: float = 0.15  # Erase up to 15% of the area


@dataclass
class TrainConfig:
    """Training configuration."""

    split_json: Path
    output_dir: Path
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LR
    num_workers: int = 0
    label_remap: dict[int, int] | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    enable_augmentation: bool = True


@dataclass
class TrainingMetrics:
    """Metrics returned from a training run.

    This allows the experiment framework to collect results programmatically.
    """

    # Final metrics from last epoch
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float

    # Best validation accuracy achieved
    best_val_acc: float

    # Test set metrics
    test_loss: float
    test_acc: float

    # Training info
    epochs_completed: int
    train_samples: int
    val_samples: int
    test_samples: int


class VideoDataset(Dataset):
    """Dataset for loading video clips from JSON split file.

    Extracts labels from filenames using pattern: *_{label}.mp4
    Supports label remapping for binary classification.
    Applies augmentations to training data.
    """

    def __init__(
        self,
        video_paths: list[str],
        label_remap: dict[int, int] | None = None,
        is_train: bool = False,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            video_paths: List of video file paths (local or gs:// URIs)
            label_remap: Optional dict to remap labels (e.g., {2: 0})
            is_train: Whether this is training data (enables augmentations)
            augmentation: Augmentation configuration
        """
        self.video_paths = video_paths  # Keep as strings to support GCS URIs
        self.label_remap = label_remap or {}
        self.is_train = is_train
        self.augmentation = augmentation or AugmentationConfig()
        self.labels: list[int] = []

        # Extract labels from filenames
        self._extract_labels()

        # Build label index mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        mode = "train" if is_train else "eval"
        logger.info(
            "Dataset ({mode}): {n} videos, {c} classes, labels={labels}, grayscale={gray}",
            mode=mode,
            n=len(self.video_paths),
            c=self.num_classes,
            labels=unique_labels,
            gray=self.augmentation.grayscale,
        )

    def _extract_labels(self) -> None:
        """Extract labels from filenames."""
        # Pattern: filename ends with _XXX.mp4 where XXX is the label
        label_pattern = re.compile(r"_(\d+)\.mp4$")

        for path in self.video_paths:
            # Extract filename from path (works for both local and GCS paths)
            filename = path.rstrip("/").split("/")[-1]
            match = label_pattern.search(filename)
            if match:
                label = int(match.group(1))
                # Apply remapping
                label = self.label_remap.get(label, label)
                self.labels.append(label)
            else:
                logger.warning("Could not extract label from: {name}", name=filename)
                self.labels.append(0)  # Default fallback

    def _to_grayscale(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Convert RGB frames to grayscale (replicated to 3 channels for model compatibility).

        Args:
            frames: List of RGB frames (H, W, 3)

        Returns:
            List of grayscale frames replicated to 3 channels (H, W, 3)
        """
        grayscale_frames = []
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Replicate to 3 channels
            gray_3ch = np.stack([gray, gray, gray], axis=-1)
            grayscale_frames.append(gray_3ch)
        return grayscale_frames

    def _apply_horizontal_flip(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply horizontal flip to all frames consistently."""
        return [cv2.flip(frame, 1) for frame in frames]

    def _apply_brightness_contrast(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply random brightness and contrast adjustment to all frames consistently."""
        cfg = self.augmentation

        # Random brightness factor: 1 ± delta
        brightness = 1.0 + random.uniform(-cfg.brightness_delta, cfg.brightness_delta)
        # Random contrast factor: 1 ± delta
        contrast = 1.0 + random.uniform(-cfg.contrast_delta, cfg.contrast_delta)

        adjusted_frames = []
        for frame in frames:
            # Standard contrast/brightness formula centered at 128
            frame_float = frame.astype(np.float32)
            frame_adjusted = contrast * (frame_float - 128.0) + 128.0 + (brightness - 1.0) * 255.0
            frame_adjusted = np.clip(frame_adjusted, 0, 255).astype(np.uint8)
            adjusted_frames.append(frame_adjusted)

        return adjusted_frames

    def _apply_gaussian_blur(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply Gaussian blur to all frames consistently."""
        cfg = self.augmentation
        sigma = random.uniform(0.1, cfg.blur_sigma_max)
        # Kernel size must be odd and at least MIN_BLUR_KERNEL
        min_kernel = 3
        kernel_size = max(int(2 * round(3 * sigma) + 1), min_kernel)

        return [cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma) for frame in frames]

    def _apply_rotation(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply small random rotation to all frames consistently."""
        cfg = self.augmentation
        angle = random.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)

        h, w = frames[0].shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_frames = []
        for frame in frames:
            rotated = cv2.warpAffine(
                frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )
            rotated_frames.append(rotated)

        return rotated_frames

    def _apply_scale_crop(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply random scale and crop to all frames consistently."""
        cfg = self.augmentation
        scale = random.uniform(cfg.scale_range[0], cfg.scale_range[1])

        h, w = frames[0].shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Calculate crop position
        if new_h > h:
            # Zoomed in: need to crop
            top = random.randint(0, new_h - h)
            left = random.randint(0, new_w - w)
        else:
            top, left = 0, 0

        scaled_frames = []
        for frame in frames:
            # Scale
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if new_h >= h and new_w >= w:
                # Crop to original size
                cropped = scaled[top : top + h, left : left + w]
            else:
                # Pad if scaled down (though scale_range typically >= 0.9)
                cropped = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
            scaled_frames.append(cropped)

        return scaled_frames

    def _apply_cutout(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply random cutout (erasing) to all frames at the same location."""
        cfg = self.augmentation
        h, w = frames[0].shape[:2]

        # Calculate cutout size
        cutout_h = int(h * np.sqrt(cfg.cutout_ratio))
        cutout_w = int(w * np.sqrt(cfg.cutout_ratio))

        # Random position
        top = random.randint(0, h - cutout_h)
        left = random.randint(0, w - cutout_w)

        cutout_frames = []
        for frame in frames:
            frame_copy = frame.copy()
            # Fill with mean gray value (128 for grayscale, or ImageNet mean for RGB)
            frame_copy[top : top + cutout_h, left : left + cutout_w] = 128
            cutout_frames.append(frame_copy)

        return cutout_frames

    def _temporal_sample_with_offset(self, frames: list[np.ndarray]) -> np.ndarray:
        """Sample NUM_FRAMES with random temporal offset for training.

        Strategy:
        - If video > NUM_FRAMES: uniform sampling with random offset
        - If video < NUM_FRAMES: loop/repeat frames
        """
        n_frames = len(frames)
        cfg = self.augmentation

        if n_frames == NUM_FRAMES:
            return np.stack(frames)

        if n_frames > NUM_FRAMES:
            if self.is_train and cfg.temporal_offset:
                # Random offset for training
                max_offset = n_frames - NUM_FRAMES
                offset = random.randint(0, max_offset)
                # Sample NUM_FRAMES starting from offset
                sample_indices = np.linspace(offset, offset + NUM_FRAMES - 1, NUM_FRAMES, dtype=int)
                sample_indices = np.clip(sample_indices, 0, n_frames - 1)
            else:
                # Uniform sampling for validation/test
                sample_indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int)
            return np.stack([frames[i] for i in sample_indices])

        # Loop frames for short videos
        loop_indices = [i % n_frames for i in range(NUM_FRAMES)]
        return np.stack([frames[i] for i in loop_indices])

    def _apply_train_augmentations(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply training-only augmentations to frames."""
        cfg = self.augmentation

        # Horizontal flip
        if random.random() < cfg.horizontal_flip_prob:
            frames = self._apply_horizontal_flip(frames)

        # Brightness and contrast
        if cfg.brightness_delta > 0 or cfg.contrast_delta > 0:
            frames = self._apply_brightness_contrast(frames)

        # Scale and crop
        if cfg.scale_range != (1.0, 1.0):
            frames = self._apply_scale_crop(frames)

        # Gaussian blur
        if random.random() < cfg.blur_prob:
            frames = self._apply_gaussian_blur(frames)

        # Small rotation
        if random.random() < cfg.rotation_prob:
            frames = self._apply_rotation(frames)

        # Cutout
        if random.random() < cfg.cutout_prob:
            frames = self._apply_cutout(frames)

        return frames

    def _read_local_video(self, local_path: Path) -> list[np.ndarray] | None:
        """Read raw RGB frames from a local video file."""
        cap = cv2.VideoCapture(str(local_path))
        if not cap.isOpened():
            logger.warning("Failed to open video: {path}", path=local_path)
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
            logger.warning("No frames extracted from: {path}", path=local_path)
            return None

        return frames

    def _read_video_frames(self, video_path: str) -> list[np.ndarray] | None:
        """Read raw RGB frames from video file (local or GCS).

        Args:
            video_path: Local path or GCS URI (gs://bucket/path)

        Returns:
            List of RGB frames, or None if failed
        """
        if video_path.startswith("gs://"):
            # Download from GCS to temp file
            temp_path = download_gcs_to_temp(video_path)
            if temp_path is None:
                return None
            try:
                return self._read_local_video(temp_path)
            finally:
                # Clean up temp file
                with contextlib.suppress(OSError):
                    temp_path.unlink()
        else:
            return self._read_local_video(Path(video_path))

    def _load_video_frames(self, video_path: str) -> np.ndarray | None:
        """Load and preprocess video frames with augmentations.

        Args:
            video_path: Path to video file

        Returns:
            Array of shape (NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3) normalized
        """
        frames = self._read_video_frames(video_path)
        if frames is None:
            return None

        cfg = self.augmentation

        # Always apply grayscale (train + val + test)
        if cfg.grayscale:
            frames = self._to_grayscale(frames)

        # Apply train-only augmentations
        if self.is_train:
            frames = self._apply_train_augmentations(frames)

        # Resize all frames to target size
        frames = [
            cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ]

        # Temporal sampling (with offset for training)
        frames_array = self._temporal_sample_with_offset(frames)

        # Normalize
        frames_normalized = frames_array.astype(np.float32) / 255.0
        frames_normalized = (frames_normalized - IMAGENET_MEAN) / IMAGENET_STD

        result: np.ndarray = frames_normalized.astype(np.float32)
        return result

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """Get a video sample."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        try:
            frames = self._load_video_frames(video_path)
            if frames is None:
                frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
        except Exception as e:
            logger.warning("Error loading {path}: {err}", path=video_path, err=e)
            frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        # Convert from (T, H, W, C) to (T, C, H, W) for VideoMAE
        frames_tensor: torch.Tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        # Extract filename from path (works for both local and GCS paths)
        filename = video_path.rstrip("/").split("/")[-1]

        return {
            "pixel_values": frames_tensor,
            "label": label_idx,
            "filename": filename,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate function to batch video samples."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    filenames = [item["filename"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "filenames": filenames,
    }


def load_split_data(
    split_json: Path,
    label_remap: dict[int, int] | None = None,
    augmentation: AugmentationConfig | None = None,
    enable_augmentation: bool = True,
) -> tuple[VideoDataset, VideoDataset, VideoDataset]:
    """Load train/val/test datasets from split JSON.

    Args:
        split_json: Path to dataset_split.json
        label_remap: Optional label remapping dict
        augmentation: Augmentation configuration
        enable_augmentation: Whether to enable train augmentations

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    with split_json.open() as f:
        split_data = json.load(f)

    train_paths = split_data["train"]
    val_paths = split_data["val"]
    test_paths = split_data["test"]

    logger.info(
        "Loading splits: train={t}, val={v}, test={te}",
        t=len(train_paths),
        v=len(val_paths),
        te=len(test_paths),
    )

    aug_cfg = augmentation or AugmentationConfig()

    # Training dataset with augmentations
    train_dataset = VideoDataset(
        train_paths,
        label_remap,
        is_train=enable_augmentation,  # Only enable train augmentations if flag is set
        augmentation=aug_cfg,
    )

    # Validation and test datasets without train augmentations (but grayscale still applies)
    val_dataset = VideoDataset(
        val_paths,
        label_remap,
        is_train=False,
        augmentation=aug_cfg,
    )
    test_dataset = VideoDataset(
        test_paths,
        label_remap,
        is_train=False,
        augmentation=aug_cfg,
    )

    return train_dataset, val_dataset, test_dataset


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {device}", device=device)
    return device


def train_epoch(
    model: VideoMAEForVideoClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            logger.info(
                "Epoch {e} | Batch {b}/{total} | Loss: {loss:.4f}",
                e=epoch,
                b=batch_idx + 1,
                total=len(dataloader),
                loss=loss.item(),
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(
    model: VideoMAEForVideoClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total * 100 if total > 0 else 0.0
    return avg_loss, accuracy


@dataclass
class CheckpointInfo:
    """Information for saving a checkpoint."""

    output_dir: Path
    epoch: int
    val_accuracy: float
    is_best: bool = False


def save_checkpoint(
    model: VideoMAEForVideoClassification,
    dataset: VideoDataset,
    info: CheckpointInfo,
) -> None:
    """Save model checkpoint."""
    info.output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    checkpoint_dir = info.output_dir / f"checkpoint-epoch-{info.epoch}"
    model.save_pretrained(checkpoint_dir)

    # Save label mapping
    label_mapping = {
        "idx_to_label": dataset.idx_to_label,
        "label_to_idx": dataset.label_to_idx,
        "num_classes": dataset.num_classes,
    }
    with (checkpoint_dir / "label_mapping.json").open("w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info(
        "Saved checkpoint to {path} (val_acc={acc:.1f}%)",
        path=checkpoint_dir,
        acc=info.val_accuracy,
    )

    # Save best model
    if info.is_best:
        best_dir = info.output_dir / "best_model"
        model.save_pretrained(best_dir)
        with (best_dir / "label_mapping.json").open("w") as f:
            json.dump(label_mapping, f, indent=2)
        logger.info("Saved best model to {path}", path=best_dir)


def train(config: TrainConfig) -> TrainingMetrics:
    """Main training function.

    Args:
        config: Training configuration.

    Returns:
        TrainingMetrics containing final accuracies and losses.
    """
    device = get_device()

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_split_data(
        config.split_json,
        config.label_remap,
        config.augmentation,
        config.enable_augmentation,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    # Load model
    logger.info("Loading VideoMAE model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=train_dataset.num_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Epochs: {n}", n=config.epochs)
    logger.info("Batch size: {n}", n=config.batch_size)
    logger.info("Learning rate: {lr}", lr=config.learning_rate)
    logger.info("Train samples: {n}", n=len(train_dataset))
    logger.info("Val samples: {n}", n=len(val_dataset))
    logger.info("Test samples: {n}", n=len(test_dataset))
    logger.info("Num classes: {n}", n=train_dataset.num_classes)
    logger.info("Augmentation enabled: {aug}", aug=config.enable_augmentation)
    logger.info("Grayscale: {gray}", gray=config.augmentation.grayscale)
    logger.info("=" * 60)

    best_val_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        logger.info("=" * 60)
        logger.info("EPOCH {e}/{total}", e=epoch, total=config.epochs)
        logger.info("=" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(
            "Train - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
            loss=train_loss,
            acc=train_acc,
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            "Val   - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
            loss=val_loss,
            acc=val_acc,
        )

        # Save checkpoint
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc

        checkpoint_info = CheckpointInfo(
            output_dir=config.output_dir,
            epoch=epoch,
            val_accuracy=val_acc,
            is_best=is_best,
        )
        save_checkpoint(model, train_dataset, checkpoint_info)

    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(
        "Test  - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
        loss=test_loss,
        acc=test_acc,
    )

    logger.success("Training complete! Best val accuracy: {acc:.1f}%", acc=best_val_accuracy)

    # Return metrics for programmatic access
    return TrainingMetrics(
        final_train_loss=train_loss,
        final_train_acc=train_acc,
        final_val_loss=val_loss,
        final_val_acc=val_acc,
        best_val_acc=best_val_accuracy,
        test_loss=test_loss,
        test_acc=test_acc,
        epochs_completed=config.epochs,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        test_samples=len(test_dataset),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VideoMAE for running pattern classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to dataset split JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--remap-labels",
        type=str,
        help="Remap labels, e.g., '2:0' to map label 2 to 0",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable training augmentations (grayscale still applied)",
    )
    parser.add_argument(
        "--no-grayscale",
        action="store_true",
        help="Disable grayscale conversion (keep RGB)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate split JSON
    if not args.split_json.exists():
        logger.error("Split JSON not found: {path}", path=args.split_json)
        sys.exit(1)

    # Parse label remapping
    label_remap: dict[int, int] | None = None
    if args.remap_labels:
        label_remap = {}
        for mapping in args.remap_labels.split(","):
            parts = mapping.strip().split(":")
            if len(parts) == 2:  # noqa: PLR2004
                label_remap[int(parts[0])] = int(parts[1])
        logger.info("Label remapping: {remap}", remap=label_remap)

    # Build augmentation config
    aug_config = AugmentationConfig(
        grayscale=not args.no_grayscale,
    )

    config = TrainConfig(
        split_json=args.split_json,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        label_remap=label_remap,
        augmentation=aug_config,
        enable_augmentation=not args.no_augmentation,
    )

    train(config)


if __name__ == "__main__":
    main()
