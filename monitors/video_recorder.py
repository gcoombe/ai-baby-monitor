"""
Video recording module for storing video footage from the baby monitor.
"""
import cv2
import os
import logging
from datetime import datetime
from collections import deque
import numpy as np
from monitors.video_annotator import VideoAnnotator

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
        self.save_annotated = storage_config.get('save_annotated', True)
        self.save_raw = storage_config.get('save_raw', False)
        self.pre_motion_buffer_size = storage_config.get('pre_motion_buffer', 5)
        self.post_motion_duration = storage_config.get('post_motion_duration', 10)

        # Camera settings
        self.resolution = tuple(camera_config.get('resolution', [640, 480]))
        self.fps = camera_config.get('fps', 10)

        # Recording state
        self.video_writer_annotated = None
        self.video_writer_raw = None
        self.recording = False
        self.current_segment_start = None
        self.current_recording_path_annotated = None
        self.current_recording_path_raw = None
        self.pre_motion_frames = deque(maxlen=self.pre_motion_buffer_size * self.fps)
        self.post_motion_frames_remaining = 0
        
        # Current frame annotation data
        self.current_frame_data = {
            'motion_level': 0,
            'contours': [],
            'is_baby_awake': False,
            'keypoints': None,
            'position': None,
            'position_confidence': 0,
            'pose_estimator': None
        }
        
        # Video annotator for adding overlays
        self.annotator = VideoAnnotator()

        # Create storage directory if enabled
        if self.enabled:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"Video recorder initialized - Mode: {self.recording_mode}, Path: {self.storage_path}")

    def process_frame(self, frame, motion_detected=False, is_baby_awake=False, 
                      last_significant_motion=None, motion_level=0, contours=None,
                      keypoints=None, position=None, position_confidence=0, pose_estimator=None):
        """
        Process a frame for recording.

        Args:
            frame: Video frame to process
            motion_detected: Whether motion was detected in this frame
            is_baby_awake: Whether baby is currently awake
            last_significant_motion: Datetime of last significant motion
            motion_level: Motion area level (pixels)
            contours: Motion contours for annotation
            keypoints: Pose estimation keypoints
            position: Detected sleeping position
            position_confidence: Confidence of position detection
            pose_estimator: Pose estimator instance for drawing keypoints
        """
        if not self.enabled:
            return

        now = datetime.now()
        
        # Store current frame annotation data
        self.current_frame_data = {
            'motion_level': motion_level,
            'contours': contours if contours is not None else [],
            'is_baby_awake': is_baby_awake,
            'keypoints': keypoints,
            'position': position,
            'position_confidence': position_confidence,
            'pose_estimator': pose_estimator
        }

        # Manage pre-motion buffer for motion mode (store frame and annotation data)
        if self.recording_mode == 'motion':
            frame_data = {
                'frame': frame.copy(),
                'timestamp': now,
                'motion_level': motion_level,
                'contours': contours if contours is not None else [],
                'is_baby_awake': is_baby_awake,
                'keypoints': keypoints,
                'position': position,
                'position_confidence': position_confidence,
                'pose_estimator': pose_estimator
            }
            self.pre_motion_frames.append(frame_data)

        # Determine if we should be recording
        should_record = self._should_record(motion_detected, is_baby_awake, last_significant_motion)

        # Start recording if needed
        if not self.recording and should_record:
            self._start_recording(now)

            # Write pre-motion buffer frames if in motion mode
            if self.recording_mode == 'motion' and len(self.pre_motion_frames) > 0:
                logger.info(f"Writing {len(self.pre_motion_frames)} pre-motion buffer frames")
                for frame_data in self.pre_motion_frames:
                    self._write_frame(frame_data['frame'], frame_data=frame_data)

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
        if self.video_writer_annotated is not None or self.video_writer_raw is not None:
            self._stop_recording()

        if start_time is None:
            start_time = datetime.now()

        # Generate base filename with timestamp
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        
        # Get codec with fallback support
        fourcc = self._get_working_codec()
        
        # Create annotated video writer if enabled
        if self.save_annotated:
            filename_annotated = f"recording_{timestamp}_annotated.mp4"
            self.current_recording_path_annotated = os.path.join(self.storage_path, filename_annotated)
            
            self.video_writer_annotated = cv2.VideoWriter(
                self.current_recording_path_annotated,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if self.video_writer_annotated.isOpened():
                logger.info(f"Started annotated recording: {self.current_recording_path_annotated}")
            else:
                logger.error(f"Failed to start annotated video recording: {self.current_recording_path_annotated}")
                self.video_writer_annotated = None
        
        # Create raw video writer if enabled
        if self.save_raw:
            filename_raw = f"recording_{timestamp}_raw.mp4"
            self.current_recording_path_raw = os.path.join(self.storage_path, filename_raw)
            
            self.video_writer_raw = cv2.VideoWriter(
                self.current_recording_path_raw,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if self.video_writer_raw.isOpened():
                logger.info(f"Started raw recording: {self.current_recording_path_raw}")
            else:
                logger.error(f"Failed to start raw video recording: {self.current_recording_path_raw}")
                self.video_writer_raw = None
        
        # Set recording state if at least one writer was opened successfully
        if (self.save_annotated and self.video_writer_annotated) or (self.save_raw and self.video_writer_raw):
            self.recording = True
            self.current_segment_start = start_time
        else:
            logger.error("Failed to start any video recording")
            self._stop_recording()

    def _stop_recording(self):
        """Stop current video recording."""
        if self.video_writer_annotated is not None:
            self.video_writer_annotated.release()
            self.video_writer_annotated = None
            logger.info(f"Stopped annotated recording: {self.current_recording_path_annotated}")
            self.current_recording_path_annotated = None
        
        if self.video_writer_raw is not None:
            self.video_writer_raw.release()
            self.video_writer_raw = None
            logger.info(f"Stopped raw recording: {self.current_recording_path_raw}")
            self.current_recording_path_raw = None
        
        self.recording = False
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
            'high': ('X264', 25)  # Use X264 for better H.264 compatibility on Raspberry Pi
        }
        codec_str, quality = quality_settings.get(self.quality, ('mp4v', 20))
        
        # Try to create codec
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        
        # Log the codec being used
        logger.info(f"Using video codec: {codec_str} (quality level: {quality})")
        
        return fourcc, quality
    
    def _get_working_codec(self):
        """
        Get a working codec for the system with fallback support.
        Tries multiple codecs and returns the first one that works.
        
        Returns:
            fourcc: Working codec fourcc code
        """
        # Define codec priority based on quality setting
        if self.quality == 'high':
            # Try H.264 variants in order of preference
            codecs_to_try = [
                ('X264', 'X264 software H.264'),
                ('H264', 'H264 codec'),
                ('avc1', 'avc1 H.264'),
                ('mp4v', 'MPEG-4 fallback')
            ]
        elif self.quality == 'medium':
            codecs_to_try = [
                ('mp4v', 'MPEG-4 Part 2'),
                ('X264', 'X264 software H.264 fallback')
            ]
        else:  # low
            codecs_to_try = [
                ('mp4v', 'MPEG-4 Part 2')
            ]
        
        # Try each codec with a test writer
        for codec_str, description in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                
                # Test if codec works by creating a temporary writer
                test_path = os.path.join(self.storage_path, '.codec_test.mp4')
                test_writer = cv2.VideoWriter(
                    test_path,
                    fourcc,
                    self.fps,
                    self.resolution
                )
                
                if test_writer.isOpened():
                    test_writer.release()
                    # Clean up test file
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    logger.info(f"Selected codec: {codec_str} ({description})")
                    return fourcc
                else:
                    test_writer.release()
                    logger.warning(f"Codec {codec_str} ({description}) failed to open, trying next...")
                    
            except Exception as e:
                logger.warning(f"Error testing codec {codec_str}: {e}")
                continue
        
        # Ultimate fallback - mp4v should always work
        logger.warning("All preferred codecs failed, using mp4v fallback")
        return cv2.VideoWriter_fourcc(*'mp4v')

    def _add_annotations(self, frame, frame_data):
        """
        Add annotations to frame using VideoAnnotator.

        Args:
            frame: Video frame
            frame_data: Dict containing annotation data

        Returns:
            Annotated frame
        """
        return self.annotator.annotate_frame(
            frame,
            motion_level=frame_data.get('motion_level', 0),
            contours=frame_data.get('contours', []),
            is_baby_awake=frame_data.get('is_baby_awake', False),
            keypoints=frame_data.get('keypoints'),
            position=frame_data.get('position'),
            position_confidence=frame_data.get('position_confidence', 0),
            pose_estimator=frame_data.get('pose_estimator')
        )

    def _add_timestamp_overlay(self, frame):
        """
        Add timestamp overlay to frame using VideoAnnotator.

        Args:
            frame: Video frame

        Returns:
            Frame with timestamp overlay
        """
        if not self.include_timestamp:
            return frame

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return self.annotator.add_timestamp_overlay(frame, timestamp)

    def _write_frame(self, frame, frame_data=None):
        """
        Write frame to video file(s).

        Args:
            frame: Video frame to write
            frame_data: Optional dict with annotation data (motion_level, contours, etc.)
        """
        if not self.recording:
            return
        
        # Write annotated video if enabled
        if self.video_writer_annotated is not None:
            if frame_data is None:
                frame_data = self.current_frame_data
            
            # Apply annotations
            frame_annotated = self._add_annotations(frame, frame_data)
            
            # Add timestamp overlay
            frame_annotated = self._add_timestamp_overlay(frame_annotated)
            
            self.video_writer_annotated.write(frame_annotated)
        
        # Write raw video if enabled
        if self.video_writer_raw is not None:
            # Only add timestamp to raw video, no annotations
            frame_raw = frame.copy()
            frame_raw = self._add_timestamp_overlay(frame_raw)
            
            self.video_writer_raw.write(frame_raw)

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
        Get path(s) to current recording file(s).

        Returns:
            dict: Dictionary with 'annotated' and 'raw' paths (None if not recording)
        """
        return {
            'annotated': self.current_recording_path_annotated,
            'raw': self.current_recording_path_raw
        }
