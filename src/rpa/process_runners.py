"""Video preprocessing module for multi-runner feet tracking and cropping.

This module processes videos containing multiple runners and generates separate
cropped output videos for each unique runner, focusing specifically on their feet.
Uses YOLOv8-Pose for person detection/tracking and extracts stabilized foot crops.

Supports both FRONTAL and LATERAL (side) views with stride-aware ROI sizing.

Usage:
    uv run python -m rpa.process_runners --input VIDEO --output-dir DIR [OPTIONS]

CLI Options:
    --input PATH              (required) Path to input video file
    --output-dir PATH         (required) Directory for output videos

    Dynamic ROI Sizing (Stride-Aware):
    --roi-height-ratio FLOAT  Crop size as % of person's bbox height (default: 0.40)
                              Used for frontal views / distance-based zoom
    --foot-length-ratio FLOAT Foot length as fraction of body height (default: 0.15)
                              Used for geometry-based stride calculation
    --min-roi-size INT        Minimum crop size in pixels (default: 128)
                              Prevents uselessly small crops for distant runners

    The final ROI size = max(height_based, stride_based, min_roi_size)
    - height_based = bbox_height * roi_height_ratio
    - stride_based = ankle_distance + 2 * bbox_height * foot_length_ratio
    - Frontal view: height-based dominates (ankles close together)
    - Lateral view: stride-based dominates (ankles far apart)

    Vertical Positioning:
    --ankle-vertical-ratio FLOAT  Where ankles appear in crop, 0=top, 1=bottom (default: 0.35)
                                  Lower values leave more room below ankles for feet
    --side-view-y-offset-ratio FLOAT  Additional downward shift for ground contact (default: 0.1)
                                      Helps keep ground contact visible in side views
    --max-padding-ratio FLOAT     Skip frames with more than this ratio of black padding (default: 0.25)
                                  Filters out frames where feet have left the video frame

    Output:
    --output-size INT         Final output video size in pixels (default: 224)
                              Output is always square (e.g., 224x224)

    Track Filtering:
    --min-frames INT          Minimum frames to keep a track (default: 20)
                              Filters out ghost tracks / brief detections
    --conf-threshold FLOAT    YOLO detection confidence threshold (default: 0.25)

    Smoothing:
    --smoothing-window INT    Window size for center position smoothing (default: 5)
    --height-smoothing-window INT  Window size for ROI size smoothing (default: 15)
                                   Critical for stride-aware sizing to prevent zoom pumping

    Temporal Slicing (for VideoMAE training):
    --slice-len INT           Frames per training clip (default: 32)
    --stride INT              Sliding window stride between clips (default: 16)
                              Creates 50% overlap with default settings

    Runner Detection (filter out walkers/standers):
    --no-runner-detection     Disable runner detection, keep all tracks
    --min-ankle-variance FLOAT  Minimum ankle variance for running gait (default: 0=disabled)
    --top-n-fastest INT       Keep only top N fastest tracks (default: 0=use ratio)
    --min-speed-ratio FLOAT   Keep tracks with speed >= ratio * max_speed (default: 0.5)

    Debug:
    --visualize               Show real-time visualization during processing

Examples:
    # Basic usage with defaults (works for both frontal and lateral views)
    uv run python -m rpa.process_runners \\
        --input /path/to/video.mp4 \\
        --output-dir /path/to/output

    # Adjust foot size estimation (larger feet / more padding)
    uv run python -m rpa.process_runners \\
        --input video.mp4 --output-dir out/ \\
        --foot-length-ratio 0.18 --side-view-y-offset-ratio 0.15

    # Tighter crops for frontal view
    uv run python -m rpa.process_runners \\
        --input frontal.mp4 --output-dir out/ \\
        --roi-height-ratio 0.35 --ankle-vertical-ratio 0.30

Output Structure:
    output-dir/
    ├── {input_stem}_ID_{track_id}.mp4    # Full stabilized video per runner
    └── clips/
        └── {input_stem}_ID_{track_id}_clip_{NNN}.mp4  # Training clips
"""

import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

# YOLO pose COCO keypoint indices for ankles (0-indexed)
# Full mapping: 0=nose, 1-4=eyes/ears, 5-10=shoulders/elbows/wrists,
# 11-12=hips, 13-14=knees, 15-16=ankles
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


@dataclass
class FrameData:
    """Data for a single frame detection of a tracked person."""

    frame_idx: int
    left_ankle: tuple[float, float] | None
    right_ankle: tuple[float, float] | None
    confidence: float
    bbox_height: float  # Height of person's bounding box for dynamic ROI scaling


@dataclass
class MotionStats:
    """Motion statistics for a tracked person to distinguish runners from walkers/standers."""

    track_id: int
    avg_speed: float  # Average movement speed in pixels/frame
    ankle_variance: float  # Variance in ankle distance (high = running gait cycle)
    vertical_oscillation: float  # Variance in vertical position (runners bounce more)
    frame_count: int


@dataclass
class PreprocessorConfig:
    """Configuration for VideoPreprocessor."""

    # Dynamic ROI sizing (scale-adaptive cropping)
    roi_height_ratio: float = 0.40  # Crop size as percentage of person's bbox height
    min_roi_size: int = 128  # Minimum crop size to prevent uselessly small crops

    # Stride-aware ROI sizing (for side/lateral views)
    # When runner's stride is wide, expand crop to keep both feet in frame
    # Uses body geometry: foot length ≈ 15% of body height
    foot_length_ratio: float = 0.15  # Foot length as fraction of body height
    side_view_y_offset_ratio: float = 0.1  # Push crop down slightly for ground contact

    # Vertical positioning: where ankles should appear in the crop (0=top, 1=bottom)
    # Default 0.35 means ankles at 35% from top, leaving 65% below for feet visibility
    ankle_vertical_ratio: float = 0.35

    # Maximum allowed padding ratio - skip frames where feet are out of bounds
    # If more than this fraction of the crop would be black padding, skip the frame
    max_padding_ratio: float = 0.25

    output_size: int = 224
    min_track_frames: int = 20
    smoothing_window: int = 5  # For center position smoothing
    height_smoothing_window: int = 15  # Larger window for ROI size to prevent zoom pulsing
    conf_threshold: float = 0.25
    visualize: bool = False

    # Phase 4: Temporal slicing for VideoMAE Transformer
    slice_len: int = 16  # Number of frames per training clip
    stride: int = 16  # Step size between slices (overlap = slice_len - stride)

    # Video quality settings
    # CRF (Constant Rate Factor) for H.264 encoding: 0=lossless, 17=visually lossless, 23=default
    # Lower = better quality, larger files. Recommended: 0-17 for training data
    video_crf: int = 0  # 0 = lossless H.264

    # Runner detection settings
    # Filter out stationary/walking people, keep only active runners
    runner_detection: bool = True  # Enable runner vs walker/stander filtering
    min_ankle_variance: float = 0.0  # Minimum ankle variance to be considered running gait (0=auto)
    top_n_fastest: int = 0  # Keep only top N fastest tracks (0=keep all above threshold)
    min_speed_ratio: float = 0.5  # Keep tracks with speed >= this ratio of max speed (0-1)


class VideoPreprocessor:
    """Process video to extract per-runner feet crops with stabilization.

    This class implements a multi-pass approach:
    1. Data Collection: Iterate through video collecting all track data
    2. Output Generation: Generate stabilized cropped videos per track
    3. Temporal Slicing: Create fixed-length clips for model training

    CROP CALCULATION LOGIC:
    -----------------------
    The crop region is centered on the midpoint between both ankles.

    1. Get left_ankle (x1, y1) and right_ankle (x2, y2) coordinates
    2. If both valid: midpoint = ((x1+x2)/2, (y1+y2)/2)
    3. If only one valid: use that ankle as center
    4. If neither valid in this frame: interpolate from neighboring frames

    DYNAMIC ROI SIZING (Scale-Adaptive Cropping):
    ---------------------------------------------
    The ROI size scales based on the detected person's bounding box height.
    This ensures consistent relative foot size regardless of camera distance.

    roi_size = max(bbox_height * roi_height_ratio, min_roi_size)

    - Far runner (small bbox): smaller crop -> digital zoom when resized to 224
    - Close runner (large bbox): larger crop -> zoom out when resized to 224
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        config: PreprocessorConfig | None = None,
    ) -> None:
        """Initialize video preprocessor.

        Args:
            input_path: Path to input video
            output_dir: Directory for output videos
            config: Configuration options (uses defaults if not provided)
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.config = config or PreprocessorConfig()

        # Unpack config for convenience
        self.roi_height_ratio = self.config.roi_height_ratio
        self.min_roi_size = self.config.min_roi_size
        self.foot_length_ratio = self.config.foot_length_ratio
        self.side_view_y_offset_ratio = self.config.side_view_y_offset_ratio
        self.ankle_vertical_ratio = self.config.ankle_vertical_ratio
        self.max_padding_ratio = self.config.max_padding_ratio
        self.output_size = self.config.output_size
        self.min_track_frames = self.config.min_track_frames
        self.smoothing_window = self.config.smoothing_window
        self.height_smoothing_window = self.config.height_smoothing_window
        self.conf_threshold = self.config.conf_threshold
        self.visualize = self.config.visualize
        self.slice_len = self.config.slice_len
        self.stride = self.config.stride
        self.video_crf = self.config.video_crf

        # Runner detection config
        self.runner_detection = self.config.runner_detection
        self.min_ankle_variance = self.config.min_ankle_variance
        self.top_n_fastest = self.config.top_n_fastest
        self.min_speed_ratio = self.config.min_speed_ratio

        self.model: YOLO | None = None
        self.fps: int = 30
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.total_frames: int = 0

    def _detect_best_device(self) -> str:
        """Detect the best available device for inference.

        Returns:
            Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Load YOLO v8 pose model for keypoint detection and tracking."""
        device = self._detect_best_device()
        logger.info("Loading YOLO v8 pose model on {device}...", device=device)
        self.model = YOLO("yolov8n-pose.pt")
        self.model.to(device)
        logger.info("Model loaded successfully on {device}", device=device)

    def _setup_video_capture(self) -> cv2.VideoCapture:
        """Setup video capture and store video properties.

        Returns:
            VideoCapture instance
        """
        logger.info("Opening input video: {path}", path=self.input_path)
        cap = cv2.VideoCapture(str(self.input_path))

        if not cap.isOpened():
            msg = f"Failed to open video: {self.input_path}"
            raise RuntimeError(msg)

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video properties: {w}x{h} @ {fps}fps, {frames} frames",
            w=self.frame_width,
            h=self.frame_height,
            fps=self.fps,
            frames=self.total_frames,
        )

        return cap

    def collect_tracks(self) -> dict[int, list[FrameData]]:  # noqa: PLR0912
        """Collect tracking data for all persons across the video.

        Uses YOLO's native tracking with persist=True to maintain track IDs
        across frames.

        Returns:
            Dictionary mapping track_id to list of FrameData
        """
        if self.model is None:
            self.load_model()

        cap = self._setup_video_capture()
        tracks: dict[int, list[FrameData]] = defaultdict(list)

        logger.info("Phase 1: Collecting track data...")
        frame_idx = 0

        try:
            assert self.model is not None  # For mypy

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run tracking inference with persist=True to maintain IDs
                results = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=self.conf_threshold,
                    classes=[0],  # Only track persons
                )

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes

                    # Check if tracking IDs are available
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)  # type: ignore[union-attr]
                        keypoints_data = results[0].keypoints

                        for i, track_id in enumerate(track_ids):
                            # Extract ankle keypoints
                            if keypoints_data is None:
                                continue
                            kpts_raw = keypoints_data[i].xy.cpu().numpy()
                            conf = float(boxes.conf[i])

                            # Extract bounding box height for dynamic ROI sizing
                            # xywh format: (x_center, y_center, width, height)
                            bbox_xywh = boxes.xywh[i].cpu().numpy()
                            bbox_height = float(bbox_xywh[3])  # Height is 4th element

                            # Squeeze to (17, 2) - handles both (1, 17, 2) and (17, 2) shapes
                            kpts = np.squeeze(kpts_raw)

                            left_ankle = None
                            right_ankle = None

                            if len(kpts) > LEFT_ANKLE_IDX:
                                la = kpts[LEFT_ANKLE_IDX]
                                if la[0] > 0 and la[1] > 0:
                                    left_ankle = (float(la[0]), float(la[1]))

                            if len(kpts) > RIGHT_ANKLE_IDX:
                                ra = kpts[RIGHT_ANKLE_IDX]
                                if ra[0] > 0 and ra[1] > 0:
                                    right_ankle = (float(ra[0]), float(ra[1]))

                            frame_data = FrameData(
                                frame_idx=frame_idx,
                                left_ankle=left_ankle,
                                right_ankle=right_ankle,
                                confidence=conf,
                                bbox_height=bbox_height,
                            )
                            tracks[track_id].append(frame_data)

                frame_idx += 1

                # Progress logging
                if frame_idx % 30 == 0:
                    progress = (frame_idx / self.total_frames) * 100
                    logger.info(
                        "Collection progress: {progress:.1f}% ({current}/{total} frames)",
                        progress=progress,
                        current=frame_idx,
                        total=self.total_frames,
                    )

        finally:
            cap.release()

        logger.info(
            "Collected {n_tracks} tracks across {n_frames} frames",
            n_tracks=len(tracks),
            n_frames=frame_idx,
        )

        return dict(tracks)

    def filter_ghost_tracks(
        self, tracks: dict[int, list[FrameData]]
    ) -> dict[int, list[FrameData]]:
        """Remove tracks with fewer than min_track_frames detections.

        Args:
            tracks: Dictionary of track_id to FrameData list

        Returns:
            Filtered dictionary with only valid tracks
        """
        filtered = {}
        removed_count = 0

        for track_id, frame_data_list in tracks.items():
            if len(frame_data_list) >= self.min_track_frames:
                filtered[track_id] = frame_data_list
            else:
                removed_count += 1
                logger.debug(
                    "Removed ghost track {id} with {n} frames",
                    id=track_id,
                    n=len(frame_data_list),
                )

        logger.info(
            "Filtered {removed} ghost tracks, {kept} tracks remaining",
            removed=removed_count,
            kept=len(filtered),
        )

        return filtered

    def calculate_motion_stats(
        self, track_id: int, frame_data_list: list[FrameData]
    ) -> MotionStats:
        """Calculate motion statistics for a track to distinguish runners from walkers/standers.

        Computes:
        - avg_speed: Average movement speed (pixels/frame) - runners move faster
        - ankle_variance: Variance in ankle distance - runners have cycling gait
        - vertical_oscillation: Variance in vertical position - runners bounce more

        Args:
            track_id: Track identifier
            frame_data_list: List of frame data for this track

        Returns:
            MotionStats with computed values
        """
        min_frames_for_stats = 2  # Need at least 2 frames to calculate speed
        if len(frame_data_list) < min_frames_for_stats:
            return MotionStats(
                track_id=track_id,
                avg_speed=0.0,
                ankle_variance=0.0,
                vertical_oscillation=0.0,
                frame_count=len(frame_data_list),
            )

        # Collect positions and ankle distances
        positions: list[tuple[float, float]] = []
        ankle_distances: list[float] = []
        y_positions: list[float] = []

        for fd in frame_data_list:
            # Calculate center position from ankles
            if fd.left_ankle and fd.right_ankle:
                cx = (fd.left_ankle[0] + fd.right_ankle[0]) / 2
                cy = (fd.left_ankle[1] + fd.right_ankle[1]) / 2
                positions.append((cx, cy))
                y_positions.append(cy)

                # Ankle spread (stride indicator) - horizontal distance between ankles
                ankle_dist = abs(fd.left_ankle[0] - fd.right_ankle[0])
                ankle_distances.append(ankle_dist)
            elif fd.left_ankle:
                positions.append(fd.left_ankle)
                y_positions.append(fd.left_ankle[1])
            elif fd.right_ankle:
                positions.append(fd.right_ankle)
                y_positions.append(fd.right_ankle[1])

        # Calculate average speed (pixels/frame)
        speeds: list[float] = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            speed = (dx**2 + dy**2) ** 0.5
            speeds.append(speed)

        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Calculate ankle distance variance (high = running gait cycle)
        ankle_variance = float(np.var(ankle_distances)) if ankle_distances else 0.0

        # Calculate vertical oscillation (runners bounce more)
        vertical_oscillation = float(np.var(y_positions)) if y_positions else 0.0

        return MotionStats(
            track_id=track_id,
            avg_speed=avg_speed,
            ankle_variance=ankle_variance,
            vertical_oscillation=vertical_oscillation,
            frame_count=len(frame_data_list),
        )

    def filter_runners(
        self, tracks: dict[int, list[FrameData]]
    ) -> dict[int, list[FrameData]]:
        """Filter tracks to keep only active runners based on motion analysis.

        Uses two criteria:
        1. Ankle variance - runners have high variance due to gait cycle
        2. Speed - runners are the fastest moving tracks in the video

        The filtering strategy:
        - Calculate motion stats for all tracks
        - If top_n_fastest > 0: keep only the N fastest tracks
        - Otherwise: keep tracks with speed >= min_speed_ratio * max_speed
        - Optionally filter by minimum ankle variance

        Args:
            tracks: Dictionary of track_id to FrameData list

        Returns:
            Filtered dictionary with only runner tracks
        """
        if not tracks:
            return tracks

        # Calculate motion stats for all tracks
        all_stats: list[MotionStats] = []
        for track_id, frame_data_list in tracks.items():
            stats = self.calculate_motion_stats(track_id, frame_data_list)
            all_stats.append(stats)

        # Log stats for debugging
        logger.info("Motion statistics for {n} tracks:", n=len(all_stats))
        for stats in sorted(all_stats, key=lambda s: s.avg_speed, reverse=True):
            logger.debug(
                "  Track {id}: speed={speed:.2f} px/frame, ankle_var={avar:.2f}, "
                "vert_osc={vosc:.2f}, frames={frames}",
                id=stats.track_id,
                speed=stats.avg_speed,
                avar=stats.ankle_variance,
                vosc=stats.vertical_oscillation,
                frames=stats.frame_count,
            )

        # Find max speed for ratio-based filtering
        max_speed = max(s.avg_speed for s in all_stats) if all_stats else 0.0

        if max_speed == 0:
            logger.warning("All tracks have zero speed - keeping all tracks")
            return tracks

        # Determine which tracks to keep
        kept_track_ids: set[int] = set()

        if self.top_n_fastest > 0:
            # Keep top N fastest tracks
            sorted_by_speed = sorted(all_stats, key=lambda s: s.avg_speed, reverse=True)
            top_tracks = sorted_by_speed[: self.top_n_fastest]
            kept_track_ids = {s.track_id for s in top_tracks}
            logger.info(
                "Keeping top {n} fastest tracks (speeds: {speeds})",
                n=self.top_n_fastest,
                speeds=[f"{s.avg_speed:.2f}" for s in top_tracks],
            )
        else:
            # Keep tracks with speed >= min_speed_ratio * max_speed
            speed_threshold = self.min_speed_ratio * max_speed
            for stats in all_stats:
                if stats.avg_speed >= speed_threshold:
                    kept_track_ids.add(stats.track_id)

            logger.info(
                "Speed threshold: {thresh:.2f} px/frame ({ratio:.0%} of max {max:.2f})",
                thresh=speed_threshold,
                ratio=self.min_speed_ratio,
                max=max_speed,
            )

        # Additionally filter by ankle variance if specified
        if self.min_ankle_variance > 0:
            before_count = len(kept_track_ids)
            kept_track_ids = {
                tid
                for tid in kept_track_ids
                if any(s.track_id == tid and s.ankle_variance >= self.min_ankle_variance for s in all_stats)
            }
            logger.info(
                "Ankle variance filter (min={min:.2f}): {before} -> {after} tracks",
                min=self.min_ankle_variance,
                before=before_count,
                after=len(kept_track_ids),
            )

        # Build filtered tracks dict
        filtered = {tid: tracks[tid] for tid in kept_track_ids if tid in tracks}

        removed_count = len(tracks) - len(filtered)
        logger.info(
            "Runner detection: kept {kept} runners, filtered {removed} non-runners",
            kept=len(filtered),
            removed=removed_count,
        )

        return filtered

    def _calculate_crop_center(self, frame_data: FrameData) -> tuple[float, float] | None:
        """Calculate crop center from ankle positions.

        CROP CENTER CALCULATION:
        1. If both ankles valid: midpoint = ((x1+x2)/2, (y1+y2)/2)
        2. If only one valid: use that ankle as center
        3. If neither valid: return None (will be interpolated later)

        Args:
            frame_data: Frame detection data with ankle positions

        Returns:
            (center_x, center_y) or None if no valid ankles
        """
        left = frame_data.left_ankle
        right = frame_data.right_ankle

        if left is not None and right is not None:
            # Both ankles valid - use midpoint
            center_x = (left[0] + right[0]) / 2
            center_y = (left[1] + right[1]) / 2
            return (center_x, center_y)
        if left is not None:
            # Only left ankle valid
            return left
        if right is not None:
            # Only right ankle valid
            return right
        # No valid ankles
        return None

    def _interpolate_missing_centers(
        self, centers: list[tuple[float, float] | None]
    ) -> list[tuple[float, float]]:
        """Interpolate missing center positions from neighboring frames.

        Uses linear interpolation to fill gaps where neither ankle was detected.

        Args:
            centers: List of (center_x, center_y) or None for missing

        Returns:
            List with all positions filled in
        """
        result = list(centers)
        n = len(result)

        # Find first and last valid indices
        first_valid = next((i for i, c in enumerate(result) if c is not None), None)
        last_valid = next((i for i, c in enumerate(reversed(result)) if c is not None), None)

        if first_valid is None:
            # No valid centers at all - use frame center
            default = (self.frame_width / 2, self.frame_height * 0.8)
            return [default] * n

        last_valid = n - 1 - last_valid if last_valid is not None else n - 1

        # Fill leading Nones with first valid
        for i in range(first_valid):
            result[i] = result[first_valid]

        # Fill trailing Nones with last valid
        for i in range(last_valid + 1, n):
            result[i] = result[last_valid]

        # Interpolate interior gaps
        i = first_valid
        while i < last_valid:
            if result[i] is None:
                # Find next valid
                j = i + 1
                while j <= last_valid and result[j] is None:
                    j += 1

                # Interpolate between i-1 and j
                start = result[i - 1]
                end = result[j]
                if start is not None and end is not None:
                    for k in range(i, j):
                        t = (k - i + 1) / (j - i + 1)
                        interp_x = start[0] + t * (end[0] - start[0])
                        interp_y = start[1] + t * (end[1] - start[1])
                        result[k] = (interp_x, interp_y)

                i = j
            else:
                i += 1

        # Type assertion for mypy - all Nones should be filled
        return [(c[0], c[1]) if c else (0.0, 0.0) for c in result]

    def smooth_positions(
        self, positions: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Apply moving average smoothing to center positions.

        Args:
            positions: List of (x, y) center positions

        Returns:
            Smoothed positions list
        """
        if len(positions) < self.smoothing_window:
            return positions

        smoothed = []
        half_window = self.smoothing_window // 2

        for i in range(len(positions)):
            start = max(0, i - half_window)
            end = min(len(positions), i + half_window + 1)

            window_x = [positions[j][0] for j in range(start, end)]
            window_y = [positions[j][1] for j in range(start, end)]

            smooth_x = sum(window_x) / len(window_x)
            smooth_y = sum(window_y) / len(window_y)

            smoothed.append((smooth_x, smooth_y))

        return smoothed

    def _interpolate_heights(self, heights: list[float]) -> list[float]:
        """Interpolate height values to fill gaps.

        Similar to center interpolation, but for scalar bbox height values.
        Heights of 0 are treated as missing data.

        Args:
            heights: List of bbox heights (0 indicates missing)

        Returns:
            List with all heights filled via linear interpolation
        """
        result = list(heights)
        n = len(result)

        # Find first and last valid (non-zero) indices
        first_valid = next((i for i, h in enumerate(result) if h > 0), None)
        last_valid_rev = next((i for i, h in enumerate(reversed(result)) if h > 0), None)

        if first_valid is None:
            # No valid heights - use default based on frame height
            default_height = self.frame_height * 0.3  # Assume person is ~30% of frame
            return [default_height] * n

        last_valid = n - 1 - last_valid_rev if last_valid_rev is not None else n - 1

        # Fill leading zeros with first valid
        for i in range(first_valid):
            result[i] = result[first_valid]

        # Fill trailing zeros with last valid
        for i in range(last_valid + 1, n):
            result[i] = result[last_valid]

        # Interpolate interior gaps (zeros)
        i = first_valid
        while i < last_valid:
            if result[i] <= 0:
                # Find next valid
                j = i + 1
                while j <= last_valid and result[j] <= 0:
                    j += 1

                # Linear interpolation between i-1 and j
                start_h = result[i - 1]
                end_h = result[j]
                for k in range(i, j):
                    t = (k - i + 1) / (j - i + 1)
                    result[k] = start_h + t * (end_h - start_h)

                i = j
            else:
                i += 1

        return result

    def _smooth_heights(self, heights: list[float]) -> list[float]:
        """Apply moving average smoothing to height values.

        Uses a larger window than center smoothing to prevent zoom pulsing.

        Args:
            heights: List of bbox heights

        Returns:
            Smoothed heights list
        """
        if len(heights) < self.height_smoothing_window:
            return heights

        smoothed = []
        half_window = self.height_smoothing_window // 2

        for i in range(len(heights)):
            start = max(0, i - half_window)
            end = min(len(heights), i + half_window + 1)

            window = heights[start:end]
            smooth_h = sum(window) / len(window)
            smoothed.append(smooth_h)

        return smoothed

    def _get_roi_bounds(
        self, center: tuple[float, float], roi_size: int
    ) -> tuple[int, int, int, int]:
        """Calculate ROI bounds from center position and dynamic size.

        Args:
            center: (center_x, center_y)
            roi_size: Size of square ROI (dynamically calculated per frame)

        Returns:
            (x1, y1, x2, y2) bounds (may extend beyond frame)
        """
        half_size = roi_size // 2
        x1 = int(center[0] - half_size)
        y1 = int(center[1] - half_size)
        x2 = int(center[0] + half_size)
        y2 = int(center[1] + half_size)

        return (x1, y1, x2, y2)

    def extract_crop_with_padding(
        self, frame: np.ndarray, bounds: tuple[int, int, int, int], roi_size: int
    ) -> np.ndarray:
        """Extract crop with black padding for out-of-bounds regions.

        This ensures the crop never errors even when ROI extends beyond frame.

        Args:
            frame: Source frame
            bounds: (x1, y1, x2, y2) ROI bounds
            roi_size: Size of the square crop (dynamically calculated per frame)

        Returns:
            roi_size x roi_size cropped image with black padding if needed
        """
        x1, y1, x2, y2 = bounds
        h, w = frame.shape[:2]

        # Create black canvas with dynamic size
        canvas = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)

        # Calculate valid source region
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        # Calculate destination region on canvas
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy valid pixels
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        return canvas

    def _is_crop_valid(self, bounds: tuple[int, int, int, int], roi_size: int) -> bool:
        """Check if a crop has acceptable amount of padding (feet in frame).

        Skips frames where too much of the crop would be black padding,
        indicating the feet have left the frame.

        Args:
            bounds: (x1, y1, x2, y2) ROI bounds
            roi_size: Size of the square crop

        Returns:
            True if crop is valid (acceptable padding), False if should skip
        """
        x1, y1, x2, y2 = bounds

        # Calculate how much of the ROI is out of bounds
        # We care most about bottom padding (feet exiting frame bottom)
        bottom_overflow = max(0, y2 - self.frame_height)
        top_overflow = max(0, -y1)
        left_overflow = max(0, -x1)
        right_overflow = max(0, x2 - self.frame_width)

        total_overflow_area = (
            bottom_overflow * roi_size  # Bottom strip
            + top_overflow * roi_size  # Top strip
            + left_overflow * roi_size  # Left strip
            + right_overflow * roi_size  # Right strip
        )

        # Avoid double-counting corners (approximate)
        total_area = roi_size * roi_size
        padding_ratio = total_overflow_area / total_area

        return padding_ratio <= self.max_padding_ratio

    def _encode_video_ffmpeg(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: float,
    ) -> bool:
        """Encode frames to video using FFmpeg with high quality settings.

        Uses H.264 with configurable CRF (Constant Rate Factor):
        - CRF 0: Lossless (large files, perfect quality)
        - CRF 17: Visually lossless (smaller files, imperceptible loss)
        - CRF 23: Default (good balance)

        Args:
            frames: List of frames (numpy arrays in BGR format)
            output_path: Path to output video file
            fps: Frames per second

        Returns:
            True if encoding succeeded, False otherwise
        """
        if not frames:
            return False

        # Use a temporary directory for frame images
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write frames as temporary PNG files (lossless intermediate)
            for i, frame in enumerate(frames):
                frame_path = tmp_path / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_path), frame)

            # FFmpeg command for high-quality H.264 encoding
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", str(tmp_path / "frame_%06d.png"),
                "-c:v", "libx264",
                "-crf", str(self.video_crf),
                "-preset", "slow",  # Better compression
                "-pix_fmt", "yuv420p",  # Compatibility
                "-movflags", "+faststart",  # Web streaming
                str(output_path),
            ]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "FFmpeg encoding failed: {err}, falling back to OpenCV",
                    err=e.stderr[:200] if e.stderr else str(e),
                )
                return False
            except FileNotFoundError:
                logger.warning("FFmpeg not found, falling back to OpenCV")
                return False

        return True

    def generate_output_videos(
        self, tracks: dict[int, list[FrameData]]
    ) -> dict[int, Path]:
        """Generate output videos for each valid track.

        Args:
            tracks: Filtered dictionary of track_id to FrameData list

        Returns:
            Dictionary mapping track_id to output video path
        """
        generated_videos: dict[int, Path] = {}

        if not tracks:
            logger.warning("No valid tracks to process")
            return generated_videos

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Phase 2: Generating output videos for {n} tracks...", n=len(tracks))

        for track_id, frame_data_list in tracks.items():
            output_path = self._generate_single_track_video(track_id, frame_data_list)
            if output_path is not None:
                generated_videos[track_id] = output_path

        return generated_videos

    def _calculate_stride_aware_roi(
        self, frame_data_list: list[FrameData]
    ) -> tuple[list[int], list[tuple[float, float]], int]:
        """Calculate stride-aware ROI sizes and adjusted centers.

        For frontal views: ROI is based on bbox height (runner distance from camera)
        For lateral/side views: ROI expands to encompass stride width (feet spread)

        Args:
            frame_data_list: List of frame data for this track

        Returns:
            Tuple of (roi_sizes, adjusted_centers, stride_dominant_count)
        """
        # === CENTER PROCESSING ===
        raw_centers = [self._calculate_crop_center(fd) for fd in frame_data_list]
        centers = self._interpolate_missing_centers(raw_centers)
        smoothed_centers = self.smooth_positions(centers)

        # === STRIDE-AWARE ROI SIZE CALCULATION ===
        # Step 1: Calculate height-based sizes (for frontal views / distance)
        raw_heights = [fd.bbox_height for fd in frame_data_list]
        interpolated_heights = self._interpolate_heights(raw_heights)
        smoothed_heights = self._smooth_heights(interpolated_heights)
        height_based_sizes = [h * self.roi_height_ratio for h in smoothed_heights]

        # Step 2: Calculate stride-based sizes (for lateral/side views)
        # Uses body geometry: stride_size = ankle_distance + 2 * foot_length
        # where foot_length ≈ bbox_height * foot_length_ratio (typically 15%)
        raw_stride_widths = [
            abs(fd.left_ankle[0] - fd.right_ankle[0])
            if fd.left_ankle is not None and fd.right_ankle is not None
            else 0.0
            for fd in frame_data_list
        ]

        # Interpolate and smooth stride values
        interpolated_strides = self._interpolate_heights(raw_stride_widths)
        smoothed_strides = self._smooth_heights(interpolated_strides)

        # Geometry-based stride size: ankle_dist + 2 * foot_length
        # This automatically adapts to runner size and stride width
        stride_based_sizes = [
            stride + 2 * height * self.foot_length_ratio
            for stride, height in zip(smoothed_strides, smoothed_heights, strict=True)
        ]

        # Step 3: Final ROI size = max(height_based, stride_based, min_roi_size)
        raw_roi_sizes = [
            max(height_based, stride_based, self.min_roi_size)
            for height_based, stride_based in zip(height_based_sizes, stride_based_sizes, strict=True)
        ]

        # Step 4: Apply additional smoothing to final ROI sizes
        roi_sizes = [int(s) for s in self._smooth_heights(raw_roi_sizes)]

        # Count stride-dominant frames for logging
        stride_dominant_count = sum(
            1 for hb, sb in zip(height_based_sizes, stride_based_sizes, strict=True)
            if sb > hb and sb > self.min_roi_size
        )

        # Step 5: Apply vertical offsets
        adjusted_centers = []
        for center, roi_size in zip(smoothed_centers, roi_sizes, strict=True):
            vertical_offset = (0.5 - self.ankle_vertical_ratio) * roi_size
            side_view_offset = roi_size * self.side_view_y_offset_ratio
            adjusted_y = center[1] - vertical_offset + side_view_offset
            adjusted_centers.append((center[0], adjusted_y))

        return roi_sizes, adjusted_centers, stride_dominant_count

    def _generate_single_track_video(
        self, track_id: int, frame_data_list: list[FrameData]
    ) -> Path | None:
        """Generate output video for a single track with stride-aware dynamic ROI sizing.

        Args:
            track_id: Track identifier
            frame_data_list: List of frame data for this track

        Returns:
            Path to generated video, or None if generation failed
        """
        # Calculate stride-aware ROI sizes and adjusted centers
        roi_sizes, adjusted_centers, stride_dominant_count = self._calculate_stride_aware_roi(
            frame_data_list
        )

        # Create frame index mappings
        frame_to_center = {
            fd.frame_idx: center
            for fd, center in zip(frame_data_list, adjusted_centers, strict=True)
        }
        frame_to_roi_size = {
            fd.frame_idx: roi_size
            for fd, roi_size in zip(frame_data_list, roi_sizes, strict=True)
        }

        # Log ROI size statistics for debugging
        if roi_sizes:
            logger.debug(
                "Track {id} ROI: min={min}, max={max}, avg={avg:.1f}, stride-dominant={sd}/{total}",
                id=track_id,
                min=min(roi_sizes),
                max=max(roi_sizes),
                avg=sum(roi_sizes) / len(roi_sizes),
                sd=stride_dominant_count,
                total=len(roi_sizes),
            )

        # Setup output path
        output_path = self.output_dir / f"{self.input_path.stem}_ID_{track_id}.mp4"

        # Collect frames first, then encode with FFmpeg for quality
        collected_frames: list[np.ndarray] = []
        frames_skipped = 0

        # Re-read video and extract crops with dynamic sizing
        cap = cv2.VideoCapture(str(self.input_path))
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in frame_to_center:
                    center = frame_to_center[frame_idx]
                    roi_size = frame_to_roi_size[frame_idx]

                    # Step 6: Use dynamic ROI size and adjusted center for this frame
                    bounds = self._get_roi_bounds(center, roi_size)

                    # Step 7: Skip frames where feet are out of bounds (too much padding)
                    if not self._is_crop_valid(bounds, roi_size):
                        frames_skipped += 1
                        frame_idx += 1
                        continue

                    crop = self.extract_crop_with_padding(frame, bounds, roi_size)

                    # Resize to output size (224x224)
                    resized = cv2.resize(
                        crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR
                    )
                    collected_frames.append(resized)

                    # Optional visualization
                    if self.visualize:
                        self._visualize_frame(frame, center, bounds, track_id)

                frame_idx += 1

        finally:
            cap.release()

        # Encode collected frames with FFmpeg (high quality)
        frames_written = len(collected_frames)
        if frames_written > 0:
            success = self._encode_video_ffmpeg(collected_frames, output_path, self.fps)
            if not success:
                # Fallback to OpenCV if FFmpeg fails
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                out = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.fps,
                    (self.output_size, self.output_size),
                )
                for f in collected_frames:
                    out.write(f)
                out.release()

        if frames_skipped > 0:
            logger.debug(
                "Track {id}: skipped {skipped} frames (feet out of bounds)",
                id=track_id,
                skipped=frames_skipped,
            )

        logger.info(
            "Generated {path} with {n} frames",
            path=output_path.name,
            n=frames_written,
        )

        return output_path

    def _visualize_frame(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        bounds: tuple[int, int, int, int],
        track_id: int,
    ) -> None:
        """Display visualization of current frame with annotations.

        Args:
            frame: Source frame
            center: Crop center position
            bounds: ROI bounds
            track_id: Current track ID
        """
        vis_frame = frame.copy()

        # Draw center point
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis_frame, (cx, cy), 5, (0, 255, 0), -1)

        # Draw ROI bounds
        x1, y1, x2, y2 = bounds
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw track ID label
        cv2.putText(
            vis_frame,
            f"Track {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Resize for display
        display_height = 720
        scale = display_height / vis_frame.shape[0]
        display_width = int(vis_frame.shape[1] * scale)
        display_frame = cv2.resize(vis_frame, (display_width, display_height))

        cv2.imshow("Processing", display_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()

    def _slice_track_video(self, video_path: Path, track_id: int) -> int:
        """Slice a stabilized track video into fixed-length training clips using sliding window.

        SLIDING WINDOW LOGIC:
        ---------------------
        Given a video of N frames, slice_len=32, and stride=16:

        1. Start at frame 0, extract frames [0:32] -> clip_001
        2. Move forward by stride (16), extract frames [16:48] -> clip_002
        3. Continue until start + slice_len > N

        Example with N=100, slice_len=32, stride=16:
            - clip_001: frames [0:32]   (0-31)
            - clip_002: frames [16:48]  (16-47)
            - clip_003: frames [32:64]  (32-63)
            - clip_004: frames [48:80]  (48-79)
            - clip_005: frames [64:96]  (64-95)
            Total: 5 overlapping clips from 100 frames

        The overlap (slice_len - stride = 16 frames) provides data augmentation
        by showing the model slightly different temporal windows of the same action.

        Args:
            video_path: Path to the stabilized track video (e.g., lap_006_ID_3.mp4)
            track_id: Track identifier for naming output clips

        Returns:
            Number of clips generated
        """
        # Create clips subfolder
        clips_dir = self.output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Open the stabilized video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for slicing: {path}", path=video_path)
            return 0

        # Get video properties - use exact FPS from source
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate: discard tracks shorter than slice_len
        if total_frames < self.slice_len:
            logger.warning(
                "Track {id} has {n} frames, less than slice_len={s}. Discarding.",
                id=track_id,
                n=total_frames,
                s=self.slice_len,
            )
            cap.release()
            return 0

        # Read all frames into memory for efficient slicing
        # (avoids multiple disk reads for overlapping windows)
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Sliding window formula: num_slices = floor((N - slice_len) / stride) + 1
        num_slices = ((len(frames) - self.slice_len) // self.stride) + 1

        if num_slices <= 0:
            logger.warning(
                "Cannot create any slices from {n} frames with slice_len={s}, stride={st}",
                n=len(frames),
                s=self.slice_len,
                st=self.stride,
            )
            return 0

        # Generate base name for clips
        base_name = video_path.stem  # e.g., "lap_006_ID_3"

        clips_generated = 0

        for i in range(num_slices):
            # Calculate window boundaries
            start_frame = i * self.stride
            end_frame = start_frame + self.slice_len

            # Safety check (should never trigger due to num_slices calculation)
            if end_frame > len(frames):
                break

            clip_name = f"{base_name}_clip_{i + 1:03d}.mp4"
            clip_path = clips_dir / clip_name

            # Extract clip frames
            clip_frames = frames[start_frame:end_frame]

            # Encode with FFmpeg (high quality)
            success = self._encode_video_ffmpeg(clip_frames, clip_path, fps)
            if not success:
                # Fallback to OpenCV if FFmpeg fails
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(
                    str(clip_path),
                    fourcc,
                    fps,
                    (width, height),
                )
                for frame in clip_frames:
                    writer.write(frame)
                writer.release()

            clips_generated += 1

        quality = "lossless" if self.video_crf == 0 else f"CRF {self.video_crf}"
        logger.debug(
            "Sliced track {id}: {n} clips from {total} frames "
            "(slice_len={s}, stride={st}, overlap={o}, quality={q})",
            id=track_id,
            n=clips_generated,
            total=len(frames),
            s=self.slice_len,
            st=self.stride,
            o=self.slice_len - self.stride,
            q=quality,
        )

        return clips_generated

    def process(self) -> None:
        """Run the full video processing pipeline.

        Pipeline Phases:
            1. Data Collection: Track all persons using YOLO pose
            2. Ghost Filtering: Remove short-lived tracks
            3. Runner Detection: Filter out walkers/standers, keep only runners
            4. Video Generation: Create stabilized feet crops per track
            5. Temporal Slicing: Slice into fixed-length clips for VideoMAE
        """
        # Phase 1: Collect tracking data
        tracks = self.collect_tracks()

        # Phase 2: Filter ghost tracks
        valid_tracks = self.filter_ghost_tracks(tracks)

        # Phase 3: Runner detection - filter out walkers/standers
        if self.runner_detection:
            logger.info("Phase 3: Runner detection (filtering non-runners)...")
            valid_tracks = self.filter_runners(valid_tracks)
        else:
            logger.info("Phase 3: Runner detection disabled, keeping all tracks")

        # Phase 4: Generate stabilized output videos
        generated_videos = self.generate_output_videos(valid_tracks)

        # Phase 5: Temporal slicing for VideoMAE Transformer
        logger.info(
            "Phase 5: Slicing into {s}-frame clips with stride {st}...",
            s=self.slice_len,
            st=self.stride,
        )
        total_clips = 0
        for track_id, video_path in generated_videos.items():
            clips = self._slice_track_video(video_path, track_id)
            total_clips += clips

        if self.visualize:
            cv2.destroyAllWindows()

        logger.success(
            "Processing complete! Generated {n} track videos and {c} training clips in {dir}",
            n=len(generated_videos),
            c=total_clips,
            dir=self.output_dir,
        )


def process_video(
    video_path: Path,
    output_dir: Path,
    config: PreprocessorConfig | None = None,
) -> None:
    """Main entry point for video processing.

    Args:
        video_path: Path to input video file
        output_dir: Directory for output videos
        config: Configuration options (uses defaults if not provided)
    """
    preprocessor = VideoPreprocessor(
        input_path=video_path,
        output_dir=output_dir,
        config=config,
    )
    preprocessor.process()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process video to extract per-runner feet crops with stabilization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output videos",
    )
    parser.add_argument(
        "--roi-height-ratio",
        type=float,
        default=0.40,
        help="Crop size as percentage of person's bbox height (default: 0.40)",
    )
    parser.add_argument(
        "--min-roi-size",
        type=int,
        default=128,
        help="Minimum crop size to prevent uselessly small crops (default: 128)",
    )
    parser.add_argument(
        "--foot-length-ratio",
        type=float,
        default=0.20,
        help="Foot length as fraction of body height for stride calc (default: 0.15)",
    )
    parser.add_argument(
        "--side-view-y-offset-ratio",
        type=float,
        default=0.1,
        help="Additional downward shift for ground contact visibility (default: 0.1)",
    )
    parser.add_argument(
        "--ankle-vertical-ratio",
        type=float,
        default=0.35,
        help="Vertical position of ankles in crop (0=top, 1=bottom, default: 0.35 for feet visibility)",
    )
    parser.add_argument(
        "--max-padding-ratio",
        type=float,
        default=0.25,
        help="Skip frames with more than this ratio of black padding (default: 0.25)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=224,
        help="Final output video size (default: 224x224)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Minimum frames to keep a track (default: 20)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for center position smoothing (default: 5)",
    )
    parser.add_argument(
        "--height-smoothing-window",
        type=int,
        default=15,
        help="Window size for height smoothing to prevent zoom pulsing (default: 15)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization during processing",
    )

    # Phase 4: Temporal slicing arguments for VideoMAE
    parser.add_argument(
        "--slice-len",
        type=int,
        default=16,
        help="Number of frames per training clip (default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Sliding window stride between clips (default: 16, creates 50%% overlap)",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=0,
        help="H.264 CRF quality (0=lossless, 17=visually lossless, 23=default). Lower=better quality",
    )

    # Runner detection arguments
    parser.add_argument(
        "--no-runner-detection",
        action="store_true",
        help="Disable runner detection (keep all tracks, including walkers/standers)",
    )
    parser.add_argument(
        "--min-ankle-variance",
        type=float,
        default=0.0,
        help="Minimum ankle variance to be considered running gait (default: 0=disabled)",
    )
    parser.add_argument(
        "--top-n-fastest",
        type=int,
        default=0,
        help="Keep only top N fastest tracks (default: 0=use speed ratio instead)",
    )
    parser.add_argument(
        "--min-speed-ratio",
        type=float,
        default=0.5,
        help="Keep tracks with speed >= this ratio of max speed (default: 0.5)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error("Input file does not exist: {path}", path=args.input)
        sys.exit(1)

    if not args.input.is_file():
        logger.error("Input path is not a file: {path}", path=args.input)
        sys.exit(1)

    logger.info("Input: {input_path}", input_path=args.input)
    logger.info("Output directory: {output_dir}", output_dir=args.output_dir)

    # Create configuration from CLI args
    config = PreprocessorConfig(
        roi_height_ratio=args.roi_height_ratio,
        min_roi_size=args.min_roi_size,
        foot_length_ratio=args.foot_length_ratio,
        side_view_y_offset_ratio=args.side_view_y_offset_ratio,
        ankle_vertical_ratio=args.ankle_vertical_ratio,
        max_padding_ratio=args.max_padding_ratio,
        output_size=args.output_size,
        min_track_frames=args.min_frames,
        smoothing_window=args.smoothing_window,
        height_smoothing_window=args.height_smoothing_window,
        conf_threshold=args.conf_threshold,
        visualize=args.visualize,
        slice_len=args.slice_len,
        stride=args.stride,
        video_crf=args.video_crf,
        # Runner detection
        runner_detection=not args.no_runner_detection,
        min_ankle_variance=args.min_ankle_variance,
        top_n_fastest=args.top_n_fastest,
        min_speed_ratio=args.min_speed_ratio,
    )

    try:
        process_video(
            video_path=args.input,
            output_dir=args.output_dir,
            config=config,
        )
    except Exception as e:
        logger.exception("Error during video processing: {error}", error=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
