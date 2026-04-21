"""Video transformation module for tracking and cropping runners."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

# YOLO pose keypoint indices
LEFT_ANKLE_IDX = 13
RIGHT_ANKLE_IDX = 14


class VideoTransformer:
    """Transform video to crop runner's feet with minimal background."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        crop_size: tuple[int, int] = (400, 300),
        feet_ratio: float = 0.25,
        padding_scale: float = 1.5,
    ) -> None:
        """Initialize video transformer.

        Args:
            input_path: Path to input video
            output_path: Path to output video
            crop_size: Output crop size as (width, height) in pixels
            feet_ratio: Ratio of lower body to extract as feet (0-1, default 0.25 = bottom 25%)
            padding_scale: Scale factor for padding around feet bbox (default 1.5)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.crop_width, self.crop_height = crop_size
        self.smoothing_alpha = 0.15  # Fixed smoothing factor
        self.feet_ratio = feet_ratio
        self.padding_scale = padding_scale

        self.model: YOLO | None = None
        self.smooth_center: tuple[float, float] | None = None

    def load_model(self) -> None:
        """Load YOLO v8 pose model for keypoint detection."""
        logger.info("Loading YOLO v8 pose model...")
        self.model = YOLO("yolov8n-pose.pt")  # Using pose model for keypoint detection
        logger.info("Model loaded successfully")

    def detect_runner_feet(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        """Detect runner's feet using pose keypoints (ankles).

        Args:
            frame: Input frame

        Returns:
            Tuple of (body_bbox, feet_bbox) as ((x1, y1, x2, y2), (x1, y1, x2, y2))
            or None if no person detected

        YOLO Pose keypoint indices:
        13: left_ankle, 14: right_ankle, 15: left_foot, 16: right_foot (if available)
        """
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        # Run pose inference
        results = self.model(frame, verbose=False)

        # Get detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Get largest detection (closest person, likely the runner)
        boxes = results[0].boxes
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        largest_idx = int(areas.argmax())

        # Get full body bounding box
        bbox = boxes.xyxy[largest_idx].cpu().numpy()
        bx1, by1, bx2, by2 = map(int, bbox)
        body_bbox = (bx1, by1, bx2, by2)

        # Get pose keypoints
        keypoints = results[0].keypoints[largest_idx].xy.cpu().numpy()

        # Extract ankle/foot keypoints (indices 13-16)
        # 13: left_ankle, 14: right_ankle
        left_ankle = keypoints[LEFT_ANKLE_IDX] if len(keypoints) > LEFT_ANKLE_IDX else None
        right_ankle = keypoints[RIGHT_ANKLE_IDX] if len(keypoints) > RIGHT_ANKLE_IDX else None

        # Filter valid keypoints (confidence > 0)
        valid_points = []
        if left_ankle is not None and left_ankle[0] > 0 and left_ankle[1] > 0:
            valid_points.append(left_ankle)
        if right_ankle is not None and right_ankle[0] > 0 and right_ankle[1] > 0:
            valid_points.append(right_ankle)

        if not valid_points:
            # No valid ankle keypoints, fallback to bottom of body bbox
            logger.debug("No valid ankle keypoints, using body bbox bottom")
            body_height = by2 - by1
            feet_height = int(body_height * self.feet_ratio)
            feet_bbox = (bx1, by2 - feet_height, bx2, by2)
        else:
            # Calculate feet bbox from ankle keypoints
            points_array = np.array(valid_points)
            min_x = int(points_array[:, 0].min())
            max_x = int(points_array[:, 0].max())
            min_y = int(points_array[:, 1].min())

            # Add margin around keypoints (ankles to ground)
            feet_width = max(max_x - min_x, 50)  # Minimum 50px width
            feet_height = max(by2 - min_y + 20, 80)  # From ankles to ground + margin

            # Center the bbox horizontally on the ankles
            center_x = (min_x + max_x) / 2
            feet_x1 = int(center_x - feet_width / 2)
            feet_x2 = int(center_x + feet_width / 2)
            feet_y1 = min_y - 20  # Start slightly above ankles
            feet_y2 = by2  # Extend to body bbox bottom (ground)

            feet_bbox = (feet_x1, feet_y1, feet_x2, feet_y2)

        confidence = float(boxes.conf[largest_idx])
        fx1, fy1, fx2, fy2 = feet_bbox
        logger.debug(
            "Body: ({bx1}, {by1}, {bx2}, {by2}), Feet: ({fx1}, {fy1}, {fx2}, {fy2}), conf: {conf:.2f}",
            bx1=bx1,
            by1=by1,
            bx2=bx2,
            by2=by2,
            fx1=fx1,
            fy1=fy1,
            fx2=fx2,
            fy2=fy2,
            conf=confidence,
        )

        return body_bbox, feet_bbox

    def smooth_center_position(self, new_center: tuple[float, float]) -> tuple[float, float]:
        """Apply exponential moving average to center position for smooth tracking.

        Args:
            new_center: New center position (x, y)

        Returns:
            Smoothed center position
        """
        if self.smooth_center is None:
            self.smooth_center = new_center
            return new_center

        smooth_x = self.smoothing_alpha * new_center[0] + (1 - self.smoothing_alpha) * self.smooth_center[0]
        smooth_y = self.smoothing_alpha * new_center[1] + (1 - self.smoothing_alpha) * self.smooth_center[1]

        self.smooth_center = (smooth_x, smooth_y)
        return self.smooth_center

    def calculate_crop_region(
        self, center: tuple[float, float], frame_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Calculate crop region centered on given point.

        Args:
            center: Center point (x, y)
            frame_shape: Frame dimensions (height, width)

        Returns:
            Crop region as (x1, y1, x2, y2)
        """
        height, width = frame_shape
        center_x, center_y = center

        # Calculate crop boundaries
        half_width = self.crop_width // 2
        half_height = self.crop_height // 2

        x1 = int(center_x - half_width)
        y1 = int(center_y - half_height)
        x2 = int(center_x + half_width)
        y2 = int(center_y + half_height)

        # Handle edge cases - keep crop within frame bounds
        if x1 < 0:
            x2 = min(self.crop_width, width)
            x1 = 0
        elif x2 > width:
            x1 = max(0, width - self.crop_width)
            x2 = width

        if y1 < 0:
            y2 = min(self.crop_height, height)
            y1 = 0
        elif y2 > height:
            y1 = max(0, height - self.crop_height)
            y2 = height

        return x1, y1, x2, y2

    def _setup_video_capture(self) -> tuple[cv2.VideoCapture, int, int, int, int]:
        """Setup video capture and return video properties.

        Returns:
            Tuple of (capture, fps, total_frames, width, height)
        """
        logger.info("Opening input video: {path}", path=self.input_path)
        cap = cv2.VideoCapture(str(self.input_path))

        if not cap.isOpened():
            msg = f"Failed to open video: {self.input_path}"
            raise RuntimeError(msg)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Video properties: {w}x{h} @ {fps}fps, {frames} frames",
            w=width,
            h=height,
            fps=fps,
            frames=total_frames,
        )

        # Adjust crop size if needed
        if self.crop_width > width or self.crop_height > height:
            scale = min(width / self.crop_width, height / self.crop_height)
            self.crop_width = int(self.crop_width * scale)
            self.crop_height = int(self.crop_height * scale)
            logger.warning(
                "Adjusted crop size to {w}x{h} to fit video dimensions",
                w=self.crop_width,
                h=self.crop_height,
            )

        return cap, fps, total_frames, width, height

    def _setup_video_writer(self, fps: int) -> cv2.VideoWriter:
        """Setup video writer for output.

        Args:
            fps: Frames per second for output video

        Returns:
            VideoWriter instance
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        return cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (self.crop_width, self.crop_height),
        )

    def _get_frame_center(
        self,
        frame: np.ndarray,
        frame_count: int,
        last_detection: tuple[float, float] | None,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None]:
        """Get feet center position for current frame based on detection.

        Args:
            frame: Current frame
            frame_count: Current frame number
            last_detection: Last known detection position
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Tuple of (feet_center_x, feet_center_y, bboxes) where bboxes is (body_bbox, feet_bbox) or None
        """
        detection_result = self.detect_runner_feet(frame)

        if detection_result is not None:
            _body_bbox, feet_bbox = detection_result
            fx1, fy1, fx2, fy2 = feet_bbox
            return ((fx1 + fx2) / 2, (fy1 + fy2) / 2, detection_result)

        if last_detection is not None:
            logger.debug("No feet detection in frame {n}, using last position", n=frame_count)
            return (*last_detection, None)

        logger.warning("No feet detection in frame {n}, using frame bottom center", n=frame_count)
        return (frame_width / 2, frame_height * 0.85, None)

    def _calculate_feet_crop(
        self, feet_bbox: tuple[int, int, int, int], frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        """Calculate crop region from feet bbox with padding and aspect ratio preservation.

        Args:
            feet_bbox: Feet bounding box (x1, y1, x2, y2)
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Crop region (x1, y1, x2, y2)
        """
        fx1, fy1, fx2, fy2 = feet_bbox

        # Calculate feet bbox dimensions
        feet_width = fx2 - fx1
        feet_height = fy2 - fy1
        feet_center_x = (fx1 + fx2) / 2
        feet_center_y = (fy1 + fy2) / 2

        # Add padding around feet bbox
        padded_width = int(feet_width * self.padding_scale)
        padded_height = int(feet_height * self.padding_scale)

        # Maintain aspect ratio of output
        target_aspect = self.crop_width / self.crop_height
        current_aspect = padded_width / padded_height

        if current_aspect > target_aspect:
            # Wider than target, adjust height
            padded_height = int(padded_width / target_aspect)
        else:
            # Taller than target, adjust width
            padded_width = int(padded_height * target_aspect)

        # Calculate crop coordinates centered on feet
        x1 = int(feet_center_x - padded_width / 2)
        y1 = int(feet_center_y - padded_height / 2)
        x2 = int(feet_center_x + padded_width / 2)
        y2 = int(feet_center_y + padded_height / 2)

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        return x1, y1, x2, y2

    def process_video(self) -> None:
        """Process video and create cropped output."""
        self.load_model()

        cap, fps, total_frames, frame_width, frame_height = self._setup_video_capture()
        out = self._setup_video_writer(fps)

        logger.info("Processing video...")
        frame_count = 0
        last_detection: tuple[float, float] | None = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Get center position for this frame
                center_x, center_y, bboxes = self._get_frame_center(
                    frame, frame_count, last_detection, frame_width, frame_height
                )
                last_detection = (center_x, center_y)

                # Calculate crop region based on feet bbox with padding
                if bboxes is not None:
                    _body_bbox, feet_bbox = bboxes
                    x1, y1, x2, y2 = self._calculate_feet_crop(feet_bbox, frame_width, frame_height)
                else:
                    # No detection, use previous smooth center
                    smooth_x, smooth_y = self.smooth_center_position((center_x, center_y))
                    x1, y1, x2, y2 = self.calculate_crop_region((smooth_x, smooth_y), (frame_height, frame_width))

                # Crop and resize to output dimensions
                cropped = frame[y1:y2, x1:x2]
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    cropped = cv2.resize(cropped, (self.crop_width, self.crop_height))
                    out.write(cropped)
                else:
                    logger.warning("Invalid crop region at frame {n}", n=frame_count)

                # Progress logging
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(
                        "Progress: {progress:.1f}% ({current}/{total} frames)",
                        progress=progress,
                        current=frame_count,
                        total=total_frames,
                    )

        finally:
            cap.release()
            out.release()

        logger.info("Processed {n} frames", n=frame_count)
        logger.info("Output saved to: {path}", path=self.output_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Transform video to crop runner's feet with minimal background")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output video file (default: input_cropped.mp4)",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=400,
        help="Width of output crop in pixels (default: 400)",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=300,
        help="Height of output crop in pixels (default: 300)",
    )
    parser.add_argument(
        "--feet-ratio",
        type=float,
        default=0.25,
        help="Ratio of lower body to extract as feet (0-1, default: 0.25)",
    )
    parser.add_argument(
        "--padding-scale",
        type=float,
        default=1.5,
        help="Padding scale around feet bbox (default: 1.5)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for video transformation."""
    args = parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error("Input file does not exist: {path}", path=args.input)
        sys.exit(1)

    if not args.input.is_file():
        logger.error("Input path is not a file: {path}", path=args.input)
        sys.exit(1)

    # Determine output path
    if args.output is None:
        output_path = args.input.parent / f"{args.input.stem}_cropped{args.input.suffix}"
    else:
        output_path = args.output

    logger.info("Input: {input_path}", input_path=args.input)
    logger.info("Output: {output_path}", output_path=output_path)

    # Create transformer and process
    try:
        transformer = VideoTransformer(
            input_path=args.input,
            output_path=output_path,
            crop_size=(args.crop_width, args.crop_height),
            feet_ratio=args.feet_ratio,
            padding_scale=args.padding_scale,
        )
        transformer.process_video()
        logger.success("Video transformation completed successfully!")
    except Exception as e:
        logger.exception("Error during video transformation: {error}", error=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
