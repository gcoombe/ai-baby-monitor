"""
Video monitoring module for detecting motion and determining if baby is awake.
Uses picamera2 for Raspberry Pi camera support.
"""
import cv2
import numpy as np
import logging
from threading import Thread, Event
from datetime import datetime, timedelta
import time
from monitors.video_recorder import VideoRecorder
from picamera2 import Picamera2

# Import pose estimation
try:
    from models.pose_estimator import TF_AVAILABLE
    from models.top_view_estimator import TopViewBabyPoseEstimator
    from models.side_view_estimator import SideViewBabyPoseEstimator
except ImportError:
    TF_AVAILABLE = False
    TopViewBabyPoseEstimator = None
    SideViewBabyPoseEstimator = None

logger = logging.getLogger(__name__)


class VideoMonitor:
    """Monitors video feed for motion and baby activity"""

    def __init__(self, config, db_manager, notification_manager):
        """
        Initialize video monitor.

        Args:
            config: Configuration dictionary
            db_manager: DatabaseManager instance
            notification_manager: NotificationManager instance
        """
        self.config = config
        self.db = db_manager
        self.notifier = notification_manager

        # Camera settings
        self.device = config['camera']['device']
        self.resolution = tuple(config['camera']['resolution'])
        self.fps = config['camera']['fps']
        self.rotation = config['camera']['rotation']

        # Detection settings
        self.motion_threshold = config['detection']['video']['motion_threshold']    
        self.motion_min_area = config['detection']['video']['motion_min_area']
        self.awake_threshold = config['detection']['video']['awake_confidence_threshold']

        # Sleep tracking settings
        self.awake_confirmation_time = config['sleep_tracking']['awake_confirmation_time']
        self.sleep_confirmation_time = config['sleep_tracking']['sleep_confirmation_time']

        # State tracking
        self.camera = None
        self.running = False
        self.thread = None
        self.stop_event = Event()

        # Motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=self.motion_threshold, detectShadows=True
        )
        self.prev_frame = None

        # Activity tracking
        self.last_motion_time = None
        self.last_significant_motion = None
        self.is_baby_awake = False
        self.activity_start_time = None
        self.inactivity_start_time = None

        # Frame buffer for awake detection
        self.frame_buffer = []
        self.buffer_size = 10

        # Pose estimation
        self.pose_estimator = None
        self.use_pose_estimation = False
        self.unsafe_position_enabled = config.get('detect_unsafe_positions', {}).get('enabled', True)
        self.current_position = None
        self.current_position_confidence = 0.0
        self.last_unsafe_position_alert = None
        self.unsafe_position_alert_cooldown = 60  # seconds between alerts
        
        # Initialize pose estimator if available
        if TF_AVAILABLE and TopViewBabyPoseEstimator and SideViewBabyPoseEstimator:
            try:
                camera_view = config['camera'].get('view', 'top').lower()
                if camera_view == 'side':
                    self.pose_estimator = SideViewBabyPoseEstimator()
                    logger.info("Pose estimation enabled (Side View)")
                else:
                    self.pose_estimator = TopViewBabyPoseEstimator()
                    logger.info("Pose estimation enabled (Top View)")
                self.use_pose_estimation = True
            except Exception as e:
                logger.warning(f"Could not initialize pose estimator: {e}")
                self.use_pose_estimation = False
        else:
            logger.warning("TensorFlow not available - pose estimation disabled")

        # Initialize video recorder
        self.video_recorder = VideoRecorder(config)

    def start(self):
        """Start video monitoring in a separate thread"""
        if self.running:
            logger.warning("Video monitor already running")
            return

        logger.info("Starting video monitor with picamera2...")
        
        try:
            self.camera = Picamera2()
            
            # Configure camera
            camera_config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            
            # Give camera time to warm up
            time.sleep(2)
            
            # Try to capture a test frame
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                logger.error("picamera2 opened but failed to capture frame")
                self.camera.close()
                return False
                
            logger.info(f"Successfully opened camera with picamera2")
            logger.info(f"Camera properties: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize picamera2: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
            return False

        self.running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        logger.info("Video monitor started")
        return True

    def stop(self):
        """Stop video monitoring"""
        if not self.running:
            return

        logger.info("Stopping video monitor...")
        self.running = False
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=5)

        if self.camera:
            self.camera.stop()
            self.camera.close()

        # Stop video recorder
        self.video_recorder.stop()

        logger.info("Video monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running and not self.stop_event.is_set():
            try:
                # Capture frame from picamera2
                frame = self.camera.capture_array()

                if frame is None or frame.size == 0:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(1)
                    continue

                # Rotate frame if needed
                if self.rotation != 0:
                    frame = self._rotate_frame(frame, self.rotation)

                # Detect motion
                motion_detected, motion_level, contours = self._detect_motion(frame)

                if motion_detected:
                    self.last_motion_time = datetime.now()

                    # Determine if this is significant motion (baby stirring/awake)
                    if motion_level > self.motion_min_area * 2:
                        self.last_significant_motion = datetime.now()
                        self._handle_significant_motion(motion_level)

                # Check sleep/wake state
                self._update_sleep_state()

                # Pose estimation and position detection
                keypoints = None
                position = None
                position_confidence = 0.0
                
                if self.use_pose_estimation and self.pose_estimator:
                    try:
                        keypoints = self.pose_estimator.predict(frame)
                        position, position_confidence = self.pose_estimator.detect_sleeping_position(keypoints)
                        
                        # Update current position
                        self.current_position = position
                        self.current_position_confidence = position_confidence
                        
                        # Check for unsafe position
                        if self.unsafe_position_enabled and position != 'unknown':
                            self._handle_unsafe_position(position, position_confidence)
                    except Exception as e:
                        logger.error(f"Error in pose estimation: {e}")

                # Store frame in buffer for advanced detection
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)

                # Process frame for recording
                self.video_recorder.process_frame(
                    frame,
                    motion_detected=motion_detected,
                    is_baby_awake=self.is_baby_awake,
                    last_significant_motion=self.last_significant_motion,
                    motion_level=motion_level,
                    contours=contours,
                    keypoints=keypoints,
                    position=position,
                    position_confidence=position_confidence,
                    pose_estimator=self.pose_estimator
                )

              
            except Exception as e:
                logger.error(f"Error in video monitor loop: {e}")
                time.sleep(1)

    def _rotate_frame(self, frame, rotation):
        """Rotate frame by specified degrees"""
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _detect_motion(self, frame):
        """
        Detect motion in the frame using background subtraction.

        Args:
            frame: Current video frame

        Returns:
            tuple: (motion_detected, motion_level, contours)
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate total motion area
        total_area = sum(cv2.contourArea(contour) for contour in contours)

        motion_detected = total_area > self.motion_min_area

        return motion_detected, total_area, contours

    def _handle_significant_motion(self, motion_level):
        """
        Handle significant motion detection.

        Args:
            motion_level: Level of motion detected
        """
        # Calculate confidence based on motion level
        confidence = min(motion_level / (self.motion_min_area * 10), 1.0)

        if not self.is_baby_awake:
            # Baby might be stirring
            self.db.log_event('stirring', confidence=confidence)
            self.db.record_stir()

            # Check if we should notify
            if confidence > 0.7:
                self.notifier.notify_baby_stirring(confidence)

    def _handle_unsafe_position(self, position, confidence):
        """
        Handle unsafe sleeping position detection.

        Args:
            position: Detected sleeping position
            confidence: Confidence of detection
        """
        now = datetime.now()
        
        # Check if position is unsafe
        if self.pose_estimator.is_unsafe_position(position):
            # Log the event
            self.db.log_event('unsafe_position', confidence=confidence, 
                            metadata={'position': position})
            
            # Send notification if cooldown period has passed
            should_notify = (
                self.last_unsafe_position_alert is None or
                (now - self.last_unsafe_position_alert).total_seconds() > self.unsafe_position_alert_cooldown
            )
            
            if should_notify:
                logger.warning(f"Unsafe sleeping position detected: {position} (confidence: {confidence:.2f})")
                self.notifier.notify_unsafe_position(position, confidence)
                self.last_unsafe_position_alert = now

    def _update_sleep_state(self):
        """Update the baby's sleep/wake state based on recent activity"""
        now = datetime.now()

        # Check for recent activity
        has_recent_motion = (
            self.last_significant_motion and
            (now - self.last_significant_motion).total_seconds() < self.awake_confirmation_time
        )

        if has_recent_motion:
            # Reset inactivity timer
            self.inactivity_start_time = None

            if not self.is_baby_awake:
                # Track when activity started
                if not self.activity_start_time:
                    self.activity_start_time = now
                else:
                    # Check if activity has been sustained long enough
                    activity_duration = (now - self.activity_start_time).total_seconds()
                    if activity_duration >= self.awake_confirmation_time:
                        self._set_baby_awake(True)
            else:
                # Reset activity timer while awake
                self.activity_start_time = now
        else:
            # No recent motion
            self.activity_start_time = None

            if self.is_baby_awake:
                # Track when inactivity started
                if not self.inactivity_start_time:
                    self.inactivity_start_time = now
                else:
                    # Check if inactivity has been sustained long enough
                    inactivity_duration = (now - self.inactivity_start_time).total_seconds()
                    if inactivity_duration >= self.sleep_confirmation_time:
                        self._set_baby_awake(False)

    def _set_baby_awake(self, is_awake):
        """
        Update baby's awake state.

        Args:
            is_awake: Whether baby is awake
        """
        if self.is_baby_awake == is_awake:
            return

        self.is_baby_awake = is_awake

        if is_awake:
            # Baby woke up
            logger.info("Baby is now awake")
            self.db.log_event('awake', confidence=0.9)
            self.db.end_sleep_session()
            self.notifier.notify_baby_awake(confidence=0.9)
        else:
            # Baby fell asleep
            logger.info("Baby is now asleep")
            self.db.log_event('asleep', confidence=0.9)
            self.db.start_sleep_session()

    def get_current_frame(self):
        """Get the most recent frame from the camera"""
        if not self.frame_buffer:
            return None
        return self.frame_buffer[-1].copy()

    def get_status(self):
        """Get current monitoring status"""
        recording_paths = self.video_recorder.get_current_recording_path()
        return {
            'running': self.running,
            'baby_awake': self.is_baby_awake,
            'last_motion': self.last_motion_time.isoformat() if self.last_motion_time else None,
            'last_significant_motion': self.last_significant_motion.isoformat() if self.last_significant_motion else None,
            'recording': self.video_recorder.is_recording(),
            'recording_path_annotated': recording_paths['annotated'],
            'recording_path_raw': recording_paths['raw'],
            'pose_estimation_enabled': self.use_pose_estimation,
            'current_position': self.current_position,
            'position_confidence': self.current_position_confidence
        }
