"""Create GCS-aware dataset split with augmented training data.

This script takes an existing runner-based split and creates a new split where:
- Train: Points to augmented versions in GCS (25 versions per original video)
- Val/Test: Points to raw versions in GCS (for unbiased evaluation)

Usage:
    uv run python -m rpa.create_augmented_split \
        --original-split dataset_split.json \
        --bucket gs://rpa-dataset-nirgofman \
        --output dataset_split_gcs.json \
        --versions 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from loguru import logger

# Constants
EXPECTED_FILENAME_PARTS = 2  # base + label


def gcs_list_files(gcs_path: str, extension: str = ".mp4") -> set[str]:
    """List files in a GCS path.

    Args:
        gcs_path: GCS path (gs://bucket/prefix/)
        extension: File extension to filter by

    Returns:
        Set of full GCS paths
    """
    try:
        result = subprocess.run(
            ["gsutil", "ls", gcs_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to list {path}: {err}", path=gcs_path, err=e.stderr)
        return set()

    return {
        line.strip()
        for line in result.stdout.strip().split("\n")
        if line.strip().endswith(extension)
    }


def extract_filename(path: str) -> str:
    """Extract filename from path (local or GCS)."""
    return path.rstrip("/").split("/")[-1]


def map_to_raw_gcs_path(local_path: str, gcs_raw_prefix: str) -> str:
    """Map a local path to its GCS raw path.

    Args:
        local_path: Local file path
        gcs_raw_prefix: GCS prefix for raw files (e.g., gs://bucket/raw/)

    Returns:
        GCS path for the raw file
    """
    filename = extract_filename(local_path)
    return f"{gcs_raw_prefix}{filename}"


def map_to_augmented_paths(
    local_path: str,
    gcs_augmented_prefix: str,
    versions: int,
    available_files: set[str],
) -> list[str]:
    """Map a local path to all its augmented versions in GCS.

    Filename format:
    - Raw: {base}_{label}.mp4 (e.g., 04HN_RUN2_CAM1_lap_006_CUT_001_001.mp4)
    - Augmented: {base}_v{NNN}_{label}.mp4 (e.g., 04HN_RUN2_CAM1_lap_006_CUT_001_v001_001.mp4)

    Args:
        local_path: Local file path
        gcs_augmented_prefix: GCS prefix for augmented files
        versions: Number of augmented versions per video
        available_files: Set of available augmented files in GCS

    Returns:
        List of GCS paths for augmented versions
    """
    filename = extract_filename(local_path)
    # Split: 04HN_..._001.mp4 -> base="04HN_...", label_ext="001.mp4"
    parts = filename.rsplit("_", 1)
    if len(parts) != EXPECTED_FILENAME_PARTS:
        logger.warning("Unexpected filename format: {name}", name=filename)
        return []

    base = parts[0]
    label_ext = parts[1]  # e.g., "001.mp4"

    augmented_paths = []
    missing_versions = []
    for v in range(1, versions + 1):
        aug_name = f"{base}_v{v:03d}_{label_ext}"
        aug_path = f"{gcs_augmented_prefix}{aug_name}"
        if aug_path in available_files:
            augmented_paths.append(aug_path)
        else:
            missing_versions.append(v)

    if missing_versions:
        logger.warning(
            "Missing {missing}/{expected} versions for {name}: v{versions}",
            missing=len(missing_versions),
            expected=versions,
            name=filename,
            versions=missing_versions[:5],  # Show first 5 missing
        )

    return augmented_paths


def create_augmented_split(
    original_split_json: Path,
    gcs_bucket: str,
    output_json: Path,
    versions: int = 25,
    strict: bool = False,
) -> dict:
    """Create a GCS-aware split with augmented training data.

    Args:
        original_split_json: Path to original dataset_split.json
        gcs_bucket: GCS bucket (e.g., gs://rpa-dataset-nirgofman)
        output_json: Output path for the new split JSON
        versions: Number of augmented versions per video
        strict: If True, fail if any augmented versions are missing

    Returns:
        The new split dictionary
    """
    # Normalize bucket path
    gcs_bucket = gcs_bucket.rstrip("/")
    gcs_raw_prefix = f"{gcs_bucket}/raw/"
    gcs_augmented_prefix = f"{gcs_bucket}/augmented/"

    # Load original split
    logger.info("Loading original split: {path}", path=original_split_json)
    with original_split_json.open() as f:
        original_split = json.load(f)

    train_paths = original_split.get("train", [])
    val_paths = original_split.get("val", [])
    test_paths = original_split.get("test", [])

    logger.info(
        "Original split: train={t}, val={v}, test={te}",
        t=len(train_paths),
        v=len(val_paths),
        te=len(test_paths),
    )

    # List available augmented files in GCS
    logger.info("Listing augmented files in GCS...")
    available_augmented = gcs_list_files(gcs_augmented_prefix)
    logger.info("Found {n} augmented files in GCS", n=len(available_augmented))

    # Map train paths to augmented versions
    logger.info("Mapping train paths to augmented versions...")
    new_train_paths: list[str] = []
    expected_total = len(train_paths) * versions
    videos_with_missing = 0

    for path in train_paths:
        augmented = map_to_augmented_paths(
            path, gcs_augmented_prefix, versions, available_augmented
        )
        if len(augmented) < versions:
            videos_with_missing += 1
        new_train_paths.extend(augmented)

    # Report missing augmentations
    missing_count = expected_total - len(new_train_paths)
    if missing_count > 0:
        coverage_pct = len(new_train_paths) / expected_total * 100
        logger.warning(
            "Missing {missing} augmented videos ({pct:.1f}% coverage, {vids} source videos affected)",
            missing=missing_count,
            pct=coverage_pct,
            vids=videos_with_missing,
        )
        if strict:
            msg = f"Strict mode: {missing_count} augmented videos missing"
            raise ValueError(msg)

    # Map val/test paths to raw GCS paths
    logger.info("Mapping val/test paths to raw GCS paths...")
    new_val_paths = [map_to_raw_gcs_path(p, gcs_raw_prefix) for p in val_paths]
    new_test_paths = [map_to_raw_gcs_path(p, gcs_raw_prefix) for p in test_paths]

    # Create new split with metadata
    metadata: dict[str, str | int | dict] = {
        "source_split": str(original_split_json),
        "gcs_bucket": gcs_bucket,
        "versions_per_video": versions,
        "original_train_count": len(train_paths),
        "augmented_train_count": len(new_train_paths),
        "val_count": len(new_val_paths),
        "test_count": len(new_test_paths),
    }

    # Preserve original metadata if present
    if "metadata" in original_split:
        metadata["original_metadata"] = original_split["metadata"]

    new_split: dict[str, list[str] | dict[str, str | int | dict]] = {
        "train": sorted(new_train_paths),
        "val": sorted(new_val_paths),
        "test": sorted(new_test_paths),
        "metadata": metadata,
    }

    # Save new split
    with output_json.open("w") as f:
        json.dump(new_split, f, indent=2)

    logger.info("=" * 60)
    logger.info("GCS SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info("Train: {n} videos (augmented)", n=len(new_train_paths))
    logger.info("Val:   {n} videos (raw)", n=len(new_val_paths))
    logger.info("Test:  {n} videos (raw)", n=len(new_test_paths))
    logger.info("Output: {path}", path=output_json)
    logger.info("=" * 60)

    return new_split


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create GCS-aware dataset split with augmented training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original-split",
        type=Path,
        required=True,
        help="Path to original dataset_split.json",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket (e.g., gs://rpa-dataset-nirgofman)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for new split JSON",
    )
    parser.add_argument(
        "--versions",
        type=int,
        default=25,
        help="Number of augmented versions per video",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any augmented versions are missing",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.original_split.exists():
        logger.error("Original split not found: {path}", path=args.original_split)
        raise SystemExit(1)

    create_augmented_split(
        original_split_json=args.original_split,
        gcs_bucket=args.bucket,
        output_json=args.output,
        versions=args.versions,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
