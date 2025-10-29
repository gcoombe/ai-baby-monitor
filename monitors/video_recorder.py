"""
Video recording module for storing video footage from the baby monitor.
"""
import cv2
import os
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Handles video recording with multiple recording modes"""

    def __init__(self, config):
        """
        Initialize video recorder.

        Args:
            config: Configuration dictionary containing video_storage settings
        """
        storage_config = config.get('video_storage', {})
        camera_config = config.get('camera', {})

        # Storage settings
        self.enabled = storage_config.get('enabled', False)
        self.storage_path = storage_config.get('storage_path', 'recordings')
        self.recording_mode = storage_config.get('recording_mode', 'motion')
        self.segment_duration = storage_config.get('segment_duration', 300)
        self.quality = storage_config.get('quality', 'medium')
        self.include_timestamp = storage_config.get('include_timestamp', True)
        self.pre_motion_buffer_size = storage_config.get('pre_motion_buffer', 5)
        self.post_motion_duration = storage_config.get('post_motion_duration', 10)

        # Camera settings
        self.resolution = tuple(camera_config.get('resolution', [640, 480]))
        self.fps = camera_config.get('fps', 10)

        # Recording state
        self.video_writer = None
        self.recording = False
        self.current_segment_start = None
        self.current_recording_path = None
        self.pre_motion_frames = deque(maxlen=self.pre_motion_buffer_size * self.fps)
        self.post_motion_frames_remaining = 0

        # Create storage directory if enabled
        if self.enabled:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"Video recorder initialized - Mode: {self.recording_mode}, Path: {self.storage_path}")

    def process_frame(self, frame, motion_detected=False, is_baby_awake=False, last_significant_motion=None):
        """
        Process a frame for recording.

        Args:
            frame: Video frame to process
            motion_detected: Whether motion was detected in this frame
            is_baby_awake: Whether baby is currently awake
            last_significant_motion: Datetime of last significant motion
        """
        if not self.enabled:
            return

        now = datetime.now()

        # Manage pre-motion buffer for motion mode
        if self.recording_mode == 'motion':
            self.pre_motion_frames.append((frame.copy(), now))

        # Determine if we should be recording
        should_record = self._should_record(motion_detected, is_baby_awake, last_significant_motion)

        # Start recording if needed
        if not self.recording and should_record:
            self._start_recording(now)

            # Write pre-motion buffer frames if in motion mode
            if self.recording_mode == 'motion' and len(self.pre_motion_frames) > 0:
                logger.info(f"Writing {len(self.pre_motion_frames)} pre-motion buffer frames")
                for buffered_frame, _ in self.pre_motion_frames:
                    self._write_frame(buffered_frame)

        # Write current frame if recording
        if self.recording:
            self._write_frame(frame)
            self._check_recording_conditions(now, motion_detected, should_record)

    def _should_record(self, motion_detected, is_baby_awake, last_significant_motion):
        """
        Determine if recording should be active based on recording mode.

        Args:
            motion_detected: Whether motion was detected
            is_baby_awake: Whether baby is awake
            last_significant_motion: Datetime of last significant motion

        Returns:
            bool: Whether recording should be active
        """
        if self.recording_mode == 'continuous':
            return True
        elif self.recording_mode == 'motion':
            return motion_detected
        elif self.recording_mode == 'event':
            # Record when baby is awake or recent significant motion
            has_recent_motion = (
                last_significant_motion and
                (datetime.now() - last_significant_motion).total_seconds() < 30
            )
            return is_baby_awake or has_recent_motion

        return False

    def _check_recording_conditions(self, now, motion_detected, should_record):
        """
        Check if recording should continue or stop.

        Args:
            now: Current datetime
            motion_detected: Whether motion was detected
            should_record: Whether we should be recording
        """
        # Handle continuous and event modes - segment by duration
        if self.recording_mode in ['continuous', 'event']:
            segment_duration = (now - self.current_segment_start).total_seconds()
            if segment_duration >= self.segment_duration:
                self._stop_recording()
                # Start new segment if we should still be recording
                if should_record:
                    self._start_recording(now)

        # Handle motion mode - post-motion duration
        elif self.recording_mode == 'motion':
            if motion_detected:
                # Reset post-motion counter
                self.post_motion_frames_remaining = self.post_motion_duration * self.fps
            elif self.post_motion_frames_remaining > 0:
                # Continue recording for post-motion duration
                self.post_motion_frames_remaining -= 1
            else:
                # Stop recording after post-motion period
                self._stop_recording()

    def _start_recording(self, start_time=None):
        """
        Start a new video recording segment.

        Args:
            start_time: Start time for the recording (defaults to now)
        """
        if self.video_writer is not None:
            self._stop_recording()

        if start_time is None:
            start_time = datetime.now()

        # Generate filename with timestamp
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"recording_{timestamp}.mp4"
        self.current_recording_path = os.path.join(self.storage_path, filename)

        # Get codec and create video writer
        fourcc, _ = self._get_video_codec()
        self.video_writer = cv2.VideoWriter(
            self.current_recording_path,
            fourcc,
            self.fps,
            self.resolution
        )

        if self.video_writer.isOpened():
            self.recording = True
            self.current_segment_start = start_time
            logger.info(f"Started recording: {self.current_recording_path}")
        else:
            logger.error(f"Failed to start video recording: {self.current_recording_path}")
            self.video_writer = None

    def _stop_recording(self):
        """Stop current video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            logger.info(f"Stopped recording: {self.current_recording_path}")
            self.current_recording_path = None
            self.current_segment_start = None

    def _get_video_codec(self):
        """
        Get video codec and quality settings.

        Returns:
            tuple: (fourcc codec, quality level)
        """
        quality_settings = {
            'low': ('mp4v', 15),
            'medium': ('mp4v', 20),
            'high': ('avc1', 25)
        }
        codec_str, quality = quality_settings.get(self.quality, ('mp4v', 20))
        return cv2.VideoWriter_fourcc(*codec_str), quality

    def _add_timestamp_overlay(self, frame):
        """
        Add timestamp overlay to frame.

        Args:
            frame: Video frame

        Returns:
            Frame with timestamp overlay
        """
        if not self.include_timestamp:
            return frame

        frame_copy = frame.copy()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add semi-transparent background for text
        cv2.rectangle(frame_copy, (5, 5), (250, 35), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (5, 5), (250, 35), (255, 255, 255), 1)

        # Add timestamp text
        cv2.putText(
            frame_copy,
            timestamp,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame_copy

    def _write_frame(self, frame):
        """
        Write frame to video file.

        Args:
            frame: Video frame to write
        """
        if self.video_writer is not None and self.recording:
            frame_to_write = self._add_timestamp_overlay(frame)
            self.video_writer.write(frame_to_write)

    def stop(self):
        """Stop recording and cleanup."""
        if self.recording:
            self._stop_recording()

    def is_recording(self):
        """
        Check if currently recording.

        Returns:
            bool: True if recording, False otherwise
        """
        return self.recording

    def get_current_recording_path(self):
        """
        Get path to current recording file.

        Returns:
            str: Path to current recording or None
        """
        return self.current_recording_path
