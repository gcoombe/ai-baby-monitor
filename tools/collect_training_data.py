#!/usr/bin/env python3
"""
Interactive tool to collect training data for custom activity classification model.

Run this while observing your baby to capture labeled frames for training.
Collect 200-500 images per activity for best results.
"""

import cv2
import os
from datetime import datetime
from pathlib import Path
import argparse

class DataCollector:
    """Collect labeled video frames for training"""

    ACTIVITIES = {
        '1': 'sleeping',
        '2': 'awake_calm',
        '3': 'awake_active',
        '4': 'crying'
    }

    ACTIVITY_COLORS = {
        'sleeping': (100, 200, 255),
        'awake_calm': (100, 255, 100),
        'awake_active': (50, 255, 255),
        'crying': (50, 50, 255)
    }

    def __init__(self, save_dir='training_data', camera_index=0):
        """
        Initialize data collector.

        Args:
            save_dir: Directory to save training data
            camera_index: Camera device index
        """
        self.save_dir = Path(save_dir)

        # Create directories for each activity
        for activity in self.ACTIVITIES.values():
            activity_dir = self.save_dir / activity
            activity_dir.mkdir(parents=True, exist_ok=True)

        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Stats
        self.counts = {activity: 0 for activity in self.ACTIVITIES.values()}
        self.load_existing_counts()

        # Auto-capture mode
        self.auto_capture = False
        self.auto_capture_interval = 2  # seconds
        self.last_capture_time = None
        self.current_activity = None

    def load_existing_counts(self):
        """Count existing images in each directory"""
        for activity in self.ACTIVITIES.values():
            activity_dir = self.save_dir / activity
            if activity_dir.exists():
                self.counts[activity] = len(list(activity_dir.glob("*.jpg")))

    def draw_ui(self, frame):
        """
        Draw user interface on frame.

        Args:
            frame: Video frame to draw on

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Draw semi-transparent panel
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "Training Data Collector", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Instructions
        y = 60
        cv2.putText(frame, "Press number key to capture:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Activity options
        y = 85
        for key, activity in self.ACTIVITIES.items():
            count = self.counts[activity]
            color = self.ACTIVITY_COLORS[activity]
            text = f"{key}={activity} ({count})"

            # Highlight if in auto-capture mode for this activity
            if self.auto_capture and self.current_activity == activity:
                cv2.rectangle(frame, (5, y-15), (300, y+5), color, 2)

            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20

        # Additional controls
        y = 145
        controls = "A=Auto-capture | Q=Quit"
        cv2.putText(frame, controls, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Auto-capture indicator
        if self.auto_capture:
            cv2.putText(frame, f"AUTO-CAPTURE: {self.current_activity}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def save_frame(self, frame, activity):
        """
        Save frame to disk.

        Args:
            frame: Frame to save
            activity: Activity label
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.save_dir / activity / f"{timestamp}.jpg"

        cv2.imwrite(str(filename), frame)
        self.counts[activity] += 1

        print(f"✓ Saved: {activity}/{timestamp}.jpg (total: {self.counts[activity]})")

    def collect(self):
        """Run interactive data collection"""
        print("="*60)
        print("Training Data Collection Tool")
        print("="*60)
        print("\nInstructions:")
        print("1. Press number keys (1-4) to capture frame with label")
        print("2. Press 'A' to toggle auto-capture mode")
        print("   - In auto-capture, press number to start capturing that activity")
        print("   - Press 'S' to stop auto-capture")
        print("3. Press 'Q' to quit")
        print("\nTips:")
        print("- Collect 200-500 images per activity")
        print("- Vary lighting, angles, and positions")
        print("- Include different times of day")
        print("\nCollecting to:", self.save_dir.absolute())
        print("="*60)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Auto-capture logic
            if self.auto_capture and self.current_activity:
                now = datetime.now()
                if (self.last_capture_time is None or
                    (now - self.last_capture_time).total_seconds() >= self.auto_capture_interval):
                    self.save_frame(frame, self.current_activity)
                    self.last_capture_time = now

            # Draw UI
            display_frame = self.draw_ui(frame)

            cv2.imshow('Data Collection', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('a'):
                # Toggle auto-capture
                if not self.auto_capture:
                    print("\nAuto-capture mode enabled. Press 1-4 to select activity.")
                    self.auto_capture = True
                else:
                    print("\nAuto-capture mode disabled.")
                    self.auto_capture = False
                    self.current_activity = None
                    self.last_capture_time = None

            elif key == ord('s'):
                # Stop auto-capture
                if self.auto_capture:
                    print(f"\nStopped auto-capturing {self.current_activity}")
                    self.current_activity = None
                    self.last_capture_time = None

            elif chr(key) in self.ACTIVITIES:
                activity = self.ACTIVITIES[chr(key)]

                if self.auto_capture:
                    # Set activity for auto-capture
                    self.current_activity = activity
                    self.last_capture_time = None
                    print(f"\nAuto-capturing: {activity}")
                else:
                    # Single capture
                    self.save_frame(frame, activity)

        self.camera.release()
        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "="*60)
        print("Collection Summary")
        print("="*60)
        for activity, count in self.counts.items():
            status = "✓" if count >= 200 else "⚠"
            print(f"{status} {activity}: {count} images")

        total = sum(self.counts.values())
        print(f"\nTotal images collected: {total}")

        if total > 0:
            print("\nNext steps:")
            print("1. Review images in training_data/ directory")
            print("2. Remove any bad/unclear images")
            print("3. Train model: python train_activity_classifier.py")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for activity classification")
    parser.add_argument('--output', '-o', default='training_data',
                       help='Output directory for training data')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device index')

    args = parser.parse_args()

    try:
        collector = DataCollector(args.output, args.camera)
        collector.collect()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
