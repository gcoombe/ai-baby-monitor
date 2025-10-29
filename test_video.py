#!/usr/bin/env python3
"""
Test Baby Monitor Algorithm on Stored Video

This script processes a pre-recorded video file through the baby monitoring
algorithm to test detection accuracy and visualize results.

Usage:
    python test_video.py <video_file> [options]

Examples:
    python test_video.py recording.mp4
    python test_video.py recording.mp4 --show-video
    python test_video.py recording.mp4 --save-output results.mp4 --report report.txt
"""
import cv2
import numpy as np
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import baby monitor components
try:
    from models.pose_estimator import BabyPoseEstimator, TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False
    BabyPoseEstimator = None

logger = logging.getLogger(__name__)


class VideoTester:
    """Test baby monitor algorithm on recorded video"""

    def __init__(self, video_path, config=None):
        """
        Initialize video tester.

        Args:
            video_path: Path to video file
            config: Optional configuration dict (defaults used if not provided)
        """
        self.video_path = video_path
        self.config = config or self._get_default_config()

        # Video capture
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0

        # Detection settings
        self.motion_threshold = self.config.get('motion_threshold', 25)
        self.motion_min_area = self.config.get('motion_min_area', 500)
        self.awake_confirmation_time = self.config.get('awake_confirmation_time', 30)
        self.sleep_confirmation_time = self.config.get('sleep_confirmation_time', 300)

        # Motion detector
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=self.motion_threshold, detectShadows=True
        )

        # Pose estimator (optional)
        self.pose_estimator = None
        self.use_pose_estimation = False

        # Analysis results
        self.results = {
            'total_frames': 0,
            'frames_with_motion': 0,
            'frames_with_significant_motion': 0,
            'motion_events': [],
            'sleep_wake_transitions': [],
            'positions_detected': defaultdict(int),
            'unsafe_positions': [],
            'processing_time': 0,
            'video_duration': 0,
        }

        # State tracking
        self.last_motion_time = None
        self.last_significant_motion = None
        self.is_baby_awake = False
        self.activity_start_time = None
        self.inactivity_start_time = None
        self.current_frame_idx = 0
        self.start_time = None

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'motion_threshold': 25,
            'motion_min_area': 500,
            'awake_confirmation_time': 30,
            'sleep_confirmation_time': 300,
            'rotation': 0,
        }

    def initialize(self):
        """Initialize video capture and pose estimator"""
        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.results['video_duration'] = self.total_frames / self.fps if self.fps > 0 else 0

        logger.info(f"Video loaded: {self.total_frames} frames at {self.fps:.2f} FPS")
        logger.info(f"Resolution: {self.frame_width}x{self.frame_height}")
        logger.info(f"Duration: {self.results['video_duration']:.2f} seconds")

        # Initialize pose estimator if available
        if TF_AVAILABLE and BabyPoseEstimator:
            try:
                self.pose_estimator = BabyPoseEstimator()
                self.use_pose_estimation = True
                logger.info("Pose estimation enabled")
            except Exception as e:
                logger.warning(f"Could not initialize pose estimator: {e}")
                self.use_pose_estimation = False
        else:
            logger.warning("TensorFlow not available - pose estimation disabled")

        self.start_time = datetime.now()

    def _detect_motion(self, frame):
        """
        Detect motion in frame using background subtraction.

        Args:
            frame: Current video frame

        Returns:
            tuple: (motion_detected, motion_level, fg_mask)
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

        return motion_detected, total_area, fg_mask, contours

    def _get_frame_time(self):
        """Get timestamp for current frame"""
        return timedelta(seconds=self.current_frame_idx / self.fps)

    def _update_sleep_state(self):
        """Update sleep/wake state based on recent activity"""
        frame_time = self._get_frame_time()

        # Check for recent activity
        has_recent_motion = (
            self.last_significant_motion and
            (frame_time - self.last_significant_motion).total_seconds() < self.awake_confirmation_time
        )

        if has_recent_motion:
            # Reset inactivity timer
            self.inactivity_start_time = None

            if not self.is_baby_awake:
                # Track when activity started
                if not self.activity_start_time:
                    self.activity_start_time = frame_time
                else:
                    # Check if activity has been sustained long enough
                    activity_duration = (frame_time - self.activity_start_time).total_seconds()
                    if activity_duration >= self.awake_confirmation_time:
                        self._set_baby_awake(True)
            else:
                # Reset activity timer while awake
                self.activity_start_time = frame_time
        else:
            # No recent motion
            self.activity_start_time = None

            if self.is_baby_awake:
                # Track when inactivity started
                if not self.inactivity_start_time:
                    self.inactivity_start_time = frame_time
                else:
                    # Check if inactivity has been sustained long enough
                    inactivity_duration = (frame_time - self.inactivity_start_time).total_seconds()
                    if inactivity_duration >= self.sleep_confirmation_time:
                        self._set_baby_awake(False)

    def _set_baby_awake(self, is_awake):
        """Update baby's awake state"""
        if self.is_baby_awake == is_awake:
            return

        self.is_baby_awake = is_awake
        frame_time = self._get_frame_time()

        transition = {
            'frame': self.current_frame_idx,
            'time': str(frame_time),
            'state': 'awake' if is_awake else 'asleep'
        }
        self.results['sleep_wake_transitions'].append(transition)

        logger.info(f"[{frame_time}] Baby is now {'awake' if is_awake else 'asleep'}")

    def _process_frame(self, frame, annotate=False):
        """
        Process a single frame through the monitoring algorithm.

        Args:
            frame: Video frame to process
            annotate: Whether to annotate the frame with detection results

        Returns:
            Annotated frame if annotate=True, otherwise original frame
        """
        self.results['total_frames'] += 1
        frame_time = self._get_frame_time()

        # Detect motion
        motion_detected, motion_level, fg_mask, contours = self._detect_motion(frame)

        if motion_detected:
            self.results['frames_with_motion'] += 1
            self.last_motion_time = frame_time

            # Check for significant motion
            if motion_level > self.motion_min_area * 2:
                self.results['frames_with_significant_motion'] += 1
                self.last_significant_motion = frame_time

                # Record motion event
                self.results['motion_events'].append({
                    'frame': self.current_frame_idx,
                    'time': str(frame_time),
                    'level': float(motion_level)
                })

        # Update sleep state
        self._update_sleep_state()

        # Pose estimation
        position = None
        position_confidence = 0
        keypoints = None

        if self.use_pose_estimation and self.pose_estimator:
            try:
                keypoints = self.pose_estimator.predict(frame)
                position, position_confidence = self.pose_estimator.detect_sleeping_position(keypoints)

                if position != 'unknown':
                    self.results['positions_detected'][position] += 1

                if self.pose_estimator.is_unsafe_position(position):
                    self.results['unsafe_positions'].append({
                        'frame': self.current_frame_idx,
                        'time': str(frame_time),
                        'position': position,
                        'confidence': float(position_confidence)
                    })
            except Exception as e:
                logger.error(f"Error in pose estimation: {e}")

        # Annotate frame if requested
        if annotate:
            annotated = frame.copy()

            # Draw motion contours
            if motion_detected and len(contours) > 0:
                cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

            # Draw pose keypoints
            if keypoints is not None and self.pose_estimator:
                annotated = self.pose_estimator.draw_keypoints(annotated, keypoints)

            # Add text overlays
            y_offset = 30
            cv2.putText(annotated, f"Time: {frame_time}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset += 30
            state_text = "AWAKE" if self.is_baby_awake else "ASLEEP"
            state_color = (0, 0, 255) if self.is_baby_awake else (0, 255, 0)
            cv2.putText(annotated, f"State: {state_text}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

            if motion_detected:
                y_offset += 30
                cv2.putText(annotated, f"Motion: {motion_level:.0f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if position and position != 'unknown':
                y_offset += 30
                cv2.putText(annotated, f"Position: {position} ({position_confidence:.2%})",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if self.pose_estimator and self.pose_estimator.is_unsafe_position(position):
                    y_offset += 30
                    cv2.putText(annotated, "UNSAFE POSITION!", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            return annotated

        return frame

    def process_video(self, show_video=False, save_output=None, progress_interval=100):
        """
        Process entire video through the algorithm.

        Args:
            show_video: Display video as it processes
            save_output: Path to save annotated video (None to skip)
            progress_interval: Show progress every N frames
        """
        self.initialize()

        # Setup video writer if saving output
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                save_output, fourcc, self.fps,
                (self.frame_width, self.frame_height)
            )

        logger.info("Processing video...")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Apply rotation if needed
                rotation = self.config.get('rotation', 0)
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Process frame
                annotated = self._process_frame(frame, annotate=(show_video or save_output))

                # Show video
                if show_video:
                    cv2.imshow('Baby Monitor Test', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Stopped by user")
                        break
                    elif key == ord(' '):
                        # Pause
                        cv2.waitKey(0)

                # Save frame
                if video_writer:
                    video_writer.write(annotated)

                # Progress update
                self.current_frame_idx += 1
                if self.current_frame_idx % progress_interval == 0:
                    progress = (self.current_frame_idx / self.total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.current_frame_idx}/{self.total_frames} frames)")

        finally:
            # Cleanup
            self.cap.release()
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()

            # Calculate processing time
            self.results['processing_time'] = (datetime.now() - self.start_time).total_seconds()

        logger.info("Processing complete")

    def generate_report(self, output_path=None):
        """
        Generate analysis report.

        Args:
            output_path: Path to save report (None for console only)

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("BABY MONITOR VIDEO ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Video info
        report_lines.append("VIDEO INFORMATION:")
        report_lines.append(f"  File: {self.video_path}")
        report_lines.append(f"  Duration: {self.results['video_duration']:.2f} seconds ({self.results['video_duration']/60:.2f} minutes)")
        report_lines.append(f"  Total Frames: {self.results['total_frames']}")
        report_lines.append(f"  FPS: {self.fps:.2f}")
        report_lines.append(f"  Resolution: {self.frame_width}x{self.frame_height}")
        report_lines.append(f"  Processing Time: {self.results['processing_time']:.2f} seconds")
        report_lines.append("")

        # Motion analysis
        report_lines.append("MOTION DETECTION:")
        motion_percent = (self.results['frames_with_motion'] / self.results['total_frames'] * 100) if self.results['total_frames'] > 0 else 0
        report_lines.append(f"  Frames with motion: {self.results['frames_with_motion']} ({motion_percent:.1f}%)")

        significant_percent = (self.results['frames_with_significant_motion'] / self.results['total_frames'] * 100) if self.results['total_frames'] > 0 else 0
        report_lines.append(f"  Frames with significant motion: {self.results['frames_with_significant_motion']} ({significant_percent:.1f}%)")
        report_lines.append(f"  Total motion events: {len(self.results['motion_events'])}")
        report_lines.append("")

        # Sleep/wake analysis
        report_lines.append("SLEEP/WAKE ANALYSIS:")
        report_lines.append(f"  Total transitions: {len(self.results['sleep_wake_transitions'])}")
        if self.results['sleep_wake_transitions']:
            report_lines.append("  Transitions:")
            for trans in self.results['sleep_wake_transitions']:
                report_lines.append(f"    - Frame {trans['frame']} ({trans['time']}): {trans['state'].upper()}")
        else:
            report_lines.append("  No sleep/wake transitions detected")
        report_lines.append("")

        # Position analysis
        if self.results['positions_detected']:
            report_lines.append("SLEEPING POSITIONS:")
            total_position_frames = sum(self.results['positions_detected'].values())
            for position, count in sorted(self.results['positions_detected'].items(), key=lambda x: x[1], reverse=True):
                percent = (count / total_position_frames * 100) if total_position_frames > 0 else 0
                report_lines.append(f"  {position}: {count} frames ({percent:.1f}%)")
            report_lines.append("")

        # Unsafe positions
        if self.results['unsafe_positions']:
            report_lines.append("⚠️  UNSAFE POSITIONS DETECTED:")
            report_lines.append(f"  Total unsafe positions: {len(self.results['unsafe_positions'])}")
            for unsafe in self.results['unsafe_positions'][:10]:  # Show first 10
                report_lines.append(f"    - Frame {unsafe['frame']} ({unsafe['time']}): {unsafe['position']} (confidence: {unsafe['confidence']:.2%})")
            if len(self.results['unsafe_positions']) > 10:
                report_lines.append(f"    ... and {len(self.results['unsafe_positions']) - 10} more")
            report_lines.append("")

        # Summary
        report_lines.append("SUMMARY:")
        if self.results['frames_with_significant_motion'] > 0:
            report_lines.append("  ✓ Significant motion detected - baby appears active")
        else:
            report_lines.append("  ✓ Minimal motion - baby appears calm")

        if self.results['unsafe_positions']:
            report_lines.append("  ⚠️  ALERT: Unsafe sleeping positions detected!")
        else:
            report_lines.append("  ✓ No unsafe positions detected")

        report_lines.append("")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        # Print to console
        print(report)

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")

            # Also save JSON version
            json_path = output_path.replace('.txt', '.json')
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"JSON results saved to: {json_path}")

        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test baby monitor algorithm on recorded video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video.py recording.mp4
  python test_video.py recording.mp4 --show-video
  python test_video.py recording.mp4 --save-output results.mp4 --report report.txt
  python test_video.py recording.mp4 --rotation 180
        """
    )

    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--show-video', action='store_true',
                       help='Display video during processing (press q to quit, space to pause)')
    parser.add_argument('--save-output', metavar='FILE',
                       help='Save annotated video to file')
    parser.add_argument('--report', metavar='FILE',
                       help='Save analysis report to file (also creates JSON version)')
    parser.add_argument('--rotation', type=int, default=0, choices=[0, 90, 180, 270],
                       help='Rotate video (0, 90, 180, or 270 degrees)')
    parser.add_argument('--motion-threshold', type=int, default=25,
                       help='Motion detection sensitivity (default: 25)')
    parser.add_argument('--motion-min-area', type=int, default=500,
                       help='Minimum motion area in pixels (default: 500)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Check if video file exists
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Create config from args
    config = {
        'motion_threshold': args.motion_threshold,
        'motion_min_area': args.motion_min_area,
        'rotation': args.rotation,
        'awake_confirmation_time': 30,
        'sleep_confirmation_time': 300,
    }

    try:
        # Create tester
        tester = VideoTester(args.video, config)

        # Process video
        tester.process_video(
            show_video=args.show_video,
            save_output=args.save_output
        )

        # Generate report
        tester.generate_report(output_path=args.report)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

