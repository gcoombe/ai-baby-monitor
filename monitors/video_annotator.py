"""
Video annotation module for adding visual overlays to video frames.

This module provides a centralized VideoAnnotator class used by both the
video recorder and test video script to ensure consistent annotations.
"""
import cv2
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class VideoAnnotator:
    """Handles video frame annotations including motion, state, and pose overlays"""

    def __init__(self):
        """Initialize video annotator."""
        pass

    def annotate_frame(self, frame, motion_level=0, contours=None, is_baby_awake=False,
                      keypoints=None, position=None, position_confidence=0, 
                      pose_estimator=None, frame_time=None):
        """
        Add annotations to a video frame.

        Args:
            frame: Video frame to annotate
            motion_level: Motion area level in pixels
            contours: List of motion contours to draw
            is_baby_awake: Whether baby is currently awake
            keypoints: Pose estimation keypoints
            position: Detected sleeping position
            position_confidence: Confidence of position detection (0.0-1.0)
            pose_estimator: Pose estimator instance for drawing keypoints
            frame_time: Optional timedelta or string to display as timestamp

        Returns:
            Annotated frame copy
        """
        annotated = frame.copy()

        # Draw motion contours
        if contours and len(contours) > 0 and motion_level > 0:
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        # Draw pose keypoints if available
        if keypoints is not None and pose_estimator is not None:
            try:
                annotated = pose_estimator.draw_keypoints(annotated, keypoints)
            except Exception as e:
                logger.debug(f"Error drawing keypoints: {e}")

        # Add text overlays
        y_offset = 30

        # Frame time (if provided)
        if frame_time is not None:
            time_str = str(frame_time) if isinstance(frame_time, (timedelta, str)) else frame_time
            cv2.putText(annotated, f"Time: {time_str}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # Sleep/wake state
        state_text = "AWAKE" if is_baby_awake else "ASLEEP"
        state_color = (0, 0, 255) if is_baby_awake else (0, 255, 0)
        cv2.putText(annotated, f"State: {state_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

        # Motion level
        if motion_level > 0:
            y_offset += 30
            cv2.putText(annotated, f"Motion: {motion_level:.0f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Position information
        if position and position != 'unknown':
            y_offset += 30
            cv2.putText(annotated, f"Position: {position} ({position_confidence:.2%})",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Unsafe position warning
            if pose_estimator and hasattr(pose_estimator, 'is_unsafe_position'):
                if pose_estimator.is_unsafe_position(position):
                    y_offset += 30
                    cv2.putText(annotated, "UNSAFE POSITION!", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated

    def add_timestamp_overlay(self, frame, timestamp_str):
        """
        Add timestamp overlay to frame with semi-transparent background.

        Args:
            frame: Video frame
            timestamp_str: Formatted timestamp string to display

        Returns:
            Frame with timestamp overlay
        """
        frame_copy = frame.copy()

        # Add semi-transparent background for text
        cv2.rectangle(frame_copy, (5, 5), (250, 35), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (5, 5), (250, 35), (255, 255, 255), 1)

        # Add timestamp text
        cv2.putText(
            frame_copy,
            timestamp_str,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame_copy

