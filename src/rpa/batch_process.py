"""Batch video processing module for creating training datasets.

This module processes a directory of labeled videos using the process_runners
pipeline and consolidates all generated clips into a single output folder with
standardized naming. Supports parallel processing with multiprocessing.

Input Structure:
    tagged_data_videos/
    ├── 15VR_RUN2_CAM2/
    │   └── clips/
    │       ├── lap_017_000.mp4    # Label: 000
    │       └── lap_025_001.mp4    # Label: 001
    └── 11RN_RUN1_CAM1/
        └── clips/
            └── lap_008_002.mp4    # Label: 002

Output Structure:
    output_folder/
    ├── 15VR_RUN2_CAM2_lap_017_CUT_001_000.mp4
    ├── 15VR_RUN2_CAM2_lap_017_CUT_002_000.mp4
    ├── 15VR_RUN2_CAM2_lap_025_CUT_001_001.mp4
    └── 11RN_RUN1_CAM1_lap_008_CUT_001_002.mp4

Naming Convention:
    {camera_feature}_{lap_info}_CUT_{clip_num}_{label}.mp4

Usage:
    uv run python -m rpa.batch_process \\
        --input-dir /path/to/tagged_data_videos \\
        --output-dir /path/to/training_clips \\
        --workers 4
"""

import argparse
import multiprocessing as mp
import re
import shutil
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from rpa.process_runners import PreprocessorConfig, VideoPreprocessor

# Global flag for graceful shutdown
_shutdown_requested = False


@dataclass
class ProcessingResult:
    """Result from processing a single video."""

    video_path: Path
    clips_generated: int
    success: bool
    error_message: str | None = None


def find_labeled_videos(input_dir: Path) -> list[Path]:
    """Find all labeled video files in the input directory.

    Searches recursively for .mp4 files matching the pattern lap_XXX_YYY.mp4
    where YYY is the label.

    Args:
        input_dir: Root directory to search

    Returns:
        List of paths to labeled video files
    """
    videos: list[Path] = []
    pattern = re.compile(r"lap_\d+_[\d?]{3}\.mp4$")

    for mp4_file in input_dir.rglob("*.mp4"):
        if pattern.search(mp4_file.name):
            videos.append(mp4_file)

    # Sort for consistent processing order
    videos.sort()

    logger.info("Found {n} labeled videos in {dir}", n=len(videos), dir=input_dir)
    return videos


def parse_video_info(video_path: Path, input_dir: Path) -> dict[str, str]:
    """Parse video path to extract naming components.

    Args:
        video_path: Path to video file (e.g., .../15VR_RUN2_CAM2/clips/lap_017_000.mp4)
        input_dir: Root input directory for relative path calculation

    Returns:
        Dictionary with keys: camera_feature, lap_info, label
    """
    # Get relative path from input_dir
    rel_path = video_path.relative_to(input_dir)

    # Camera feature is the first directory component (e.g., 15VR_RUN2_CAM2)
    camera_feature = rel_path.parts[0]

    # Parse filename: lap_017_000.mp4 -> lap_info=lap_017, label=000
    filename = video_path.stem  # lap_017_000
    match = re.match(r"(lap_\d+)_([\d?]{3})$", filename)

    if match:
        lap_info = match.group(1)  # lap_017
        label = match.group(2)  # 000
    else:
        # Fallback for non-standard names
        lap_info = filename
        label = "???"

    return {
        "camera_feature": camera_feature,
        "lap_info": lap_info,
        "label": label,
    }


def _signal_handler(signum: int, _frame: object) -> None:
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.warning("Shutdown requested (signal {sig}), finishing current tasks...", sig=signum)


def _worker_init() -> None:
    """Initialize worker process - configure logging and ignore SIGINT."""
    # Workers ignore SIGINT - main process handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Reconfigure logger for worker process
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>PID {process}</cyan> | <level>{message}</level>",
        level="INFO",
    )


def process_single_video_worker(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    config_dict: dict,
) -> ProcessingResult:
    """Process a single video (worker function for multiprocessing).

    This function is designed to be called in a separate process.
    It recreates the config from a dict to avoid pickling issues.

    Args:
        video_path: Path to input video
        input_dir: Root input directory
        output_dir: Final output directory for renamed clips
        config_dict: Configuration as dictionary (for pickling)

    Returns:
        ProcessingResult with status and clip count
    """
    # Reconstruct config from dict
    config = PreprocessorConfig(**config_dict)

    info = parse_video_info(video_path, input_dir)
    logger.info(
        "Processing: {camera} / {lap} (label={label})",
        camera=info["camera_feature"],
        lap=info["lap_info"],
        label=info["label"],
    )

    # Create temporary directory for process_runners output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the video preprocessor
        preprocessor = VideoPreprocessor(
            input_path=video_path,
            output_dir=temp_path,
            config=config,
        )

        try:
            preprocessor.process()
        except Exception as e:
            error_msg = str(e)
            logger.error("Failed to process {path}: {err}", path=video_path, err=error_msg)
            return ProcessingResult(
                video_path=video_path,
                clips_generated=0,
                success=False,
                error_message=error_msg,
            )

        # Find generated clips in temp_dir/clips/
        clips_dir = temp_path / "clips"
        if not clips_dir.exists():
            logger.warning("No clips generated for {path}", path=video_path)
            return ProcessingResult(
                video_path=video_path,
                clips_generated=0,
                success=True,
                error_message=None,
            )

        # Collect and rename clips
        clips = sorted(clips_dir.glob("*.mp4"))
        clips_copied = 0

        for i, clip_path in enumerate(clips, start=1):
            # New naming: {camera}_{lap}_CUT_{num}_{label}.mp4
            new_name = (
                f"{info['camera_feature']}_{info['lap_info']}_CUT_{i:03d}_{info['label']}.mp4"
            )
            dest_path = output_dir / new_name

            shutil.copy2(clip_path, dest_path)
            clips_copied += 1
            logger.debug("  Created: {name}", name=new_name)

        logger.info(
            "Completed: {camera} / {lap} -> {n} clips",
            camera=info["camera_feature"],
            lap=info["lap_info"],
            n=clips_copied,
        )

        return ProcessingResult(
            video_path=video_path,
            clips_generated=clips_copied,
            success=True,
            error_message=None,
        )


def _process_result(
    result: ProcessingResult,
    stats: dict[str, int],
) -> None:
    """Update statistics based on processing result.

    Args:
        result: Result from processing a video
        stats: Dictionary with keys 'clips', 'processed', 'failed' to update in-place
    """
    if result.success and result.clips_generated > 0:
        stats["clips"] += result.clips_generated
        stats["processed"] += 1
    elif not result.success:
        stats["failed"] += 1
    else:
        stats["processed"] += 1  # Success but no clips


def _config_to_dict(config: PreprocessorConfig) -> dict:
    """Convert PreprocessorConfig to dict for pickling across processes."""
    return {
        "roi_height_ratio": config.roi_height_ratio,
        "min_roi_size": config.min_roi_size,
        "foot_length_ratio": config.foot_length_ratio,
        "side_view_y_offset_ratio": config.side_view_y_offset_ratio,
        "ankle_vertical_ratio": config.ankle_vertical_ratio,
        "max_padding_ratio": config.max_padding_ratio,
        "output_size": config.output_size,
        "min_track_frames": config.min_track_frames,
        "smoothing_window": config.smoothing_window,
        "height_smoothing_window": config.height_smoothing_window,
        "conf_threshold": config.conf_threshold,
        "visualize": config.visualize,
        "slice_len": config.slice_len,
        "stride": config.stride,
        "video_crf": config.video_crf,
        "runner_detection": config.runner_detection,
        "min_ankle_variance": config.min_ankle_variance,
        "top_n_fastest": config.top_n_fastest,
        "min_speed_ratio": config.min_speed_ratio,
    }


def batch_process(
    input_dir: Path,
    output_dir: Path,
    config: PreprocessorConfig,
    num_workers: int = 1,
) -> None:
    """Process all videos in input directory and consolidate clips.

    Args:
        input_dir: Directory containing labeled videos
        output_dir: Directory to store all renamed clips
        config: Preprocessor configuration
        num_workers: Number of parallel workers (default: 1 for sequential)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all labeled videos
    videos = find_labeled_videos(input_dir)

    if not videos:
        logger.error("No labeled videos found in {dir}", dir=input_dir)
        sys.exit(1)

    # Convert config to dict for pickling
    config_dict = _config_to_dict(config)

    # Statistics tracking
    stats: dict[str, int] = {"clips": 0, "processed": 0, "failed": 0}

    # Set up signal handler for graceful shutdown
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False
    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    try:
        if num_workers == 1:
            # Sequential processing (no multiprocessing overhead)
            logger.info("Processing {n} videos sequentially...", n=len(videos))
            for i, video_path in enumerate(videos, start=1):
                if _shutdown_requested:
                    logger.warning("Shutdown requested, stopping...")
                    break

                logger.info(
                    "=== [{i}/{total}] {name} ===",
                    i=i,
                    total=len(videos),
                    name=video_path.name,
                )
                result = process_single_video_worker(video_path, input_dir, output_dir, config_dict)
                _process_result(result, stats)
        else:
            # Parallel processing with ProcessPoolExecutor
            logger.info(
                "Processing {n} videos with {w} workers...",
                n=len(videos),
                w=num_workers,
            )

            # Use 'spawn' context for better compatibility (especially on macOS)
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as executor:
                # Submit all tasks
                future_to_video = {
                    executor.submit(
                        process_single_video_worker,
                        video_path,
                        input_dir,
                        output_dir,
                        config_dict,
                    ): video_path
                    for video_path in videos
                }

                # Process results as they complete
                for completed, future in enumerate(as_completed(future_to_video), start=1):
                    if _shutdown_requested:
                        logger.warning("Shutdown requested, cancelling pending tasks...")
                        # Cancel all pending futures
                        for f in future_to_video:
                            f.cancel()
                        break

                    video_path = future_to_video[future]
                    try:
                        result = future.result(timeout=1)
                        _process_result(result, stats)

                        if not result.success:
                            logger.error(
                                "[{i}/{total}] Failed: {name} - {err}",
                                i=completed,
                                total=len(videos),
                                name=video_path.name,
                                err=result.error_message,
                            )

                        # Progress update
                        if completed % 10 == 0 or completed == len(videos):
                            logger.info(
                                "Progress: {i}/{total} videos, {clips} clips so far",
                                i=completed,
                                total=len(videos),
                                clips=stats["clips"],
                            )

                    except Exception as e:
                        stats["failed"] += 1
                        logger.error(
                            "[{i}/{total}] Exception processing {name}: {err}",
                            i=completed,
                            total=len(videos),
                            name=video_path.name,
                            err=e,
                        )

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    logger.success(
        "Batch processing complete!\n"
        "  Videos processed: {processed}/{total}\n"
        "  Videos failed: {failed}\n"
        "  Total clips generated: {clips}\n"
        "  Output directory: {output}",
        processed=stats["processed"],
        total=len(videos),
        failed=stats["failed"],
        clips=stats["clips"],
        output=output_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process labeled videos and consolidate training clips"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing labeled videos (e.g., tagged_data_videos)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store all renamed clips",
    )

    # Process runners config - with your specified defaults
    parser.add_argument(
        "--roi-height-ratio",
        type=float,
        default=0.50,
        help="Crop size as %% of person's bbox height (default: 0.50)",
    )
    parser.add_argument(
        "--min-roi-size",
        type=int,
        default=150,
        help="Minimum crop size in pixels (default: 150)",
    )
    parser.add_argument(
        "--height-smoothing-window",
        type=int,
        default=5,
        help="Window size for ROI size smoothing (default: 5)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=40,
        help="Minimum frames to keep a track (default: 40)",
    )
    parser.add_argument(
        "--min-speed-ratio",
        type=float,
        default=0.5,
        help="Keep tracks with speed >= this ratio of max speed (default: 0.5)",
    )
    parser.add_argument(
        "--top-n-fastest",
        type=int,
        default=2,
        help="Keep only top N fastest tracks (default: 2)",
    )
    parser.add_argument(
        "--min-ankle-variance",
        type=float,
        default=400.0,
        help="Minimum ankle variance for running gait (default: 400)",
    )

    # Additional options
    parser.add_argument(
        "--slice-len",
        type=int,
        default=16,
        help="Frames per training clip (default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Sliding window stride between clips (default: 16)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=224,
        help="Output video size in pixels (default: 224)",
    )

    # Parallelization
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential processing)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error("Input directory does not exist: {path}", path=args.input_dir)
        sys.exit(1)

    if not args.input_dir.is_dir():
        logger.error("Input path is not a directory: {path}", path=args.input_dir)
        sys.exit(1)

    logger.info("Input directory: {input_dir}", input_dir=args.input_dir)
    logger.info("Output directory: {output_dir}", output_dir=args.output_dir)

    # Create configuration with specified defaults
    config = PreprocessorConfig(
        roi_height_ratio=args.roi_height_ratio,
        min_roi_size=args.min_roi_size,
        height_smoothing_window=args.height_smoothing_window,
        min_track_frames=args.min_frames,
        min_speed_ratio=args.min_speed_ratio,
        top_n_fastest=args.top_n_fastest,
        min_ankle_variance=args.min_ankle_variance,
        slice_len=args.slice_len,
        stride=args.stride,
        output_size=args.output_size,
        runner_detection=True,
    )

    logger.info("Workers: {workers}", workers=args.workers)

    try:
        batch_process(args.input_dir, args.output_dir, config, num_workers=args.workers)
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Error during batch processing: {error}", error=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
