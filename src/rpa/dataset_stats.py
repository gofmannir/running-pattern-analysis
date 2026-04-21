"""Dataset statistics and analysis module.

Parses video filenames to extract metadata and compute distributions
for class labels, runners, and camera angles.

Filename convention: {runner}_{run}_{camera}_lap_{lap}_CUT_{cut}_{label}.mp4
Example: 04HN_RUN2_CAM1_lap_006_CUT_004_001.mp4
"""

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

# Constants
SEVERE_IMBALANCE_THRESHOLD = 3
MIN_CLI_ARGS = 2


class VideoMetadata(BaseModel):
    """Parsed metadata from a video filename."""

    path: Path
    filename: str
    runner_id: str  # e.g., "04HN"
    run_number: int  # e.g., 2
    camera: str  # e.g., "CAM1"
    lap: int  # e.g., 6
    cut: int  # e.g., 4
    label: int  # e.g., 1 (from "001")

    model_config = ConfigDict(arbitrary_types_allowed=True)


def parse_filename(filepath: Path) -> VideoMetadata | None:
    """Parse video filename to extract metadata.

    Expected format: {runner}_{run}_{camera}_lap_{lap}_CUT_{cut}_{label}.mp4
    Example: 04HN_RUN2_CAM1_lap_006_CUT_004_001.mp4

    Args:
        filepath: Path to the video file

    Returns:
        VideoMetadata if parsing succeeds, None otherwise
    """
    filename = filepath.stem
    parts = filename.split("_")

    # Need at least: runner, run, camera, lap, lap_num, CUT, cut_num, label
    min_parts = 8
    if len(parts) < min_parts:
        return None

    try:
        runner_id = parts[0]  # "04HN"
        run_str = parts[1]  # "RUN2"
        camera = parts[2]  # "CAM1"
        lap_num = int(parts[4])  # Index 3 is "lap", index 4 is "006" -> 6
        cut_num = int(parts[6])  # Index 5 is "CUT", index 6 is "004" -> 4
        label = int(parts[7])  # "001" -> 1

        # Extract run number from "RUN2" -> 2
        run_number = int(run_str[3:]) if run_str.startswith("RUN") else 0

        return VideoMetadata(
            path=filepath,
            filename=filename,
            runner_id=runner_id,
            run_number=run_number,
            camera=camera,
            lap=lap_num,
            cut=cut_num,
            label=label,
        )
    except (ValueError, IndexError):
        return None


@dataclass
class DatasetStats:
    """Computed statistics for a video dataset."""

    total_files: int = 0
    parsed_files: int = 0
    failed_files: list[Path] = field(default_factory=list)

    # Distributions
    label_counts: Counter[int] = field(default_factory=Counter)
    runner_counts: Counter[str] = field(default_factory=Counter)
    camera_counts: Counter[str] = field(default_factory=Counter)

    # Cross-tabulations for detecting correlations
    label_by_runner: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    label_by_camera: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    runner_by_camera: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    # All parsed metadata
    metadata: list[VideoMetadata] = field(default_factory=list)

    def add(self, meta: VideoMetadata) -> None:
        """Add a video's metadata to the statistics."""
        self.parsed_files += 1
        self.metadata.append(meta)

        self.label_counts[meta.label] += 1
        self.runner_counts[meta.runner_id] += 1
        self.camera_counts[meta.camera] += 1

        self.label_by_runner[meta.runner_id][meta.label] += 1
        self.label_by_camera[meta.camera][meta.label] += 1
        self.runner_by_camera[meta.camera][meta.runner_id] += 1

    @property
    def unique_runners(self) -> list[str]:
        """List of unique runner IDs."""
        return sorted(self.runner_counts.keys())

    @property
    def unique_labels(self) -> list[int]:
        """List of unique labels."""
        return sorted(self.label_counts.keys())

    @property
    def unique_cameras(self) -> list[str]:
        """List of unique camera positions."""
        return sorted(self.camera_counts.keys())

    def class_imbalance_ratio(self) -> float:
        """Ratio of max to min class count. Higher = more imbalanced."""
        if not self.label_counts:
            return 0.0
        counts = list(self.label_counts.values())
        return max(counts) / min(counts) if min(counts) > 0 else float("inf")

    def runner_imbalance_ratio(self) -> float:
        """Ratio of max to min runner sample count."""
        if not self.runner_counts:
            return 0.0
        counts = list(self.runner_counts.values())
        return max(counts) / min(counts) if min(counts) > 0 else float("inf")


def compute_dataset_stats(dataset_dir: Path, pattern: str = "*.mp4") -> DatasetStats:
    """Scan a directory and compute dataset statistics.

    Args:
        dataset_dir: Path to directory containing video files
        pattern: Glob pattern for video files

    Returns:
        DatasetStats with all computed distributions
    """
    stats = DatasetStats()

    files = sorted(dataset_dir.glob(pattern))
    stats.total_files = len(files)

    for filepath in files:
        meta = parse_filename(filepath)
        if meta is not None:
            stats.add(meta)
        else:
            stats.failed_files.append(filepath)

    return stats


def print_stats_summary(stats: DatasetStats) -> None:
    """Print a human-readable summary of dataset statistics."""
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)

    logger.info(f"Total files: {stats.total_files}")
    logger.info(f"Successfully parsed: {stats.parsed_files}")
    logger.info(f"Failed to parse: {len(stats.failed_files)}")

    logger.info("")
    logger.info(f"Unique runners: {len(stats.unique_runners)}")
    logger.info(f"Unique labels: {stats.unique_labels}")
    logger.info(f"Unique cameras: {stats.unique_cameras}")

    logger.info("")
    logger.info("CLASS DISTRIBUTION:")
    for label in stats.unique_labels:
        count = stats.label_counts[label]
        pct = count / stats.parsed_files * 100 if stats.parsed_files > 0 else 0
        logger.info(f"  Class {label}: {count:5d} ({pct:5.1f}%)")

    imbalance = stats.class_imbalance_ratio()
    if imbalance > SEVERE_IMBALANCE_THRESHOLD:
        logger.warning(f"  [WARNING] Class imbalance ratio: {imbalance:.1f}x (>3x is severe)")
    else:
        logger.info(f"  Class imbalance ratio: {imbalance:.1f}x")

    logger.info("")
    logger.info("RUNNER DISTRIBUTION:")
    for runner in stats.unique_runners:
        count = stats.runner_counts[runner]
        pct = count / stats.parsed_files * 100 if stats.parsed_files > 0 else 0
        logger.info(f"  {runner}: {count:5d} ({pct:5.1f}%)")

    logger.info("")
    logger.info("CAMERA DISTRIBUTION:")
    for cam in stats.unique_cameras:
        count = stats.camera_counts[cam]
        pct = count / stats.parsed_files * 100 if stats.parsed_files > 0 else 0
        logger.info(f"  {cam}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < MIN_CLI_ARGS:
        logger.error("Usage: python -m rpa.dataset_stats <dataset_dir>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        logger.error("Error: {path} does not exist", path=dataset_path)
        sys.exit(1)

    stats = compute_dataset_stats(dataset_path)
    print_stats_summary(stats)
