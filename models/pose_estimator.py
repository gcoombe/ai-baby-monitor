"""
Baby pose estimation using MoveNet.

Detects baby's body keypoints to determine sleeping position and detect unsafe positions.
"""
import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Pose estimation will be disabled.")
    TF_AVAILABLE = False


class BabyPoseEstimator(ABC):
    """Estimate baby's pose using MoveNet for safety monitoring"""

    # 17 keypoints detected by MoveNet
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Keypoint indices for easier access
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, model_path='tf_models/movenet_lightning.tflite'):
        """
        Initialize pose estimator.

        Args:
            model_path: Path to MoveNet TFLite model
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.input_size = self.input_details[0]['shape'][1:3]  # [192, 192]
            logger.info(f"Pose estimator initialized with input size: {self.input_size}")

        except Exception as e:
            logger.error(f"Failed to load pose estimation model: {e}")
            raise

    def preprocess(self, frame):
        """
        Preprocess frame for model input.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Preprocessed input for model
        """
        # Resize to model input size
        img = cv2.resize(frame, tuple(self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize based on model's expected input type
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        return input_data

    def predict(self, frame):
        """
        Run pose estimation on frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            keypoints: Array of shape (17, 3) with [y, x, confidence] for each keypoint
        """
        try:
            input_data = self.preprocess(frame)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            keypoints_with_scores = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0, 0]  # Shape: (17, 3) - [y, x, confidence]

            return keypoints_with_scores

        except Exception as e:
            logger.error(f"Error during pose prediction: {e}")
            return np.zeros((17, 3))

    @abstractmethod
    def detect_sleeping_position(self, keypoints, confidence_threshold=0.3):
        """
        Determine baby's sleeping position from keypoints.
        
        This method should be implemented by subclasses based on camera configuration.

        Args:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
            confidence_threshold: Minimum confidence for keypoint to be considered valid

        Returns:
            tuple: (position, confidence)
                position: 'back', 'stomach', 'side_left', 'side_right', 'unknown'
                confidence: Average confidence of detection
        """
        pass

    def is_unsafe_position(self, position):
        """
        Check if position is unsafe (face down on stomach).

        Args:
            position: Detected sleeping position

        Returns:
            bool: True if position is unsafe
        """
        return position == 'stomach'

    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.3):
        """
        Draw keypoints and skeleton on frame for visualization.

        Args:
            frame: Original BGR frame
            keypoints: Detected keypoints array (17, 3)
            confidence_threshold: Minimum confidence to draw keypoint

        Returns:
            Annotated frame with keypoints drawn
        """
        height, width = frame.shape[:2]
        annotated = frame.copy()

        # Define skeleton connections
        connections = [
            # Face
            (self.NOSE, self.LEFT_EYE),
            (self.NOSE, self.RIGHT_EYE),
            (self.LEFT_EYE, self.LEFT_EAR),
            (self.RIGHT_EYE, self.RIGHT_EAR),
            # Torso
            (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
            (self.LEFT_SHOULDER, self.LEFT_HIP),
            (self.RIGHT_SHOULDER, self.RIGHT_HIP),
            (self.LEFT_HIP, self.RIGHT_HIP),
            # Arms
            (self.LEFT_SHOULDER, self.LEFT_ELBOW),
            (self.LEFT_ELBOW, self.LEFT_WRIST),
            (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
            (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            # Legs
            (self.LEFT_HIP, self.LEFT_KNEE),
            (self.LEFT_KNEE, self.LEFT_ANKLE),
            (self.RIGHT_HIP, self.RIGHT_KNEE),
            (self.RIGHT_KNEE, self.RIGHT_ANKLE),
        ]

        # Draw connections (skeleton)
        for connection in connections:
            kp1 = keypoints[connection[0]]
            kp2 = keypoints[connection[1]]

            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                y1, x1 = int(kp1[0] * height), int(kp1[1] * width)
                y2, x2 = int(kp2[0] * height), int(kp2[1] * width)
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            y, x, conf = keypoint

            if conf > confidence_threshold:
                y_coord = int(y * height)
                x_coord = int(x * width)

                # Different colors for different body parts
                if i < 5:  # Face
                    color = (255, 0, 0)  # Blue
                elif i < 11:  # Upper body
                    color = (0, 255, 0)  # Green
                else:  # Lower body
                    color = (0, 0, 255)  # Red

                cv2.circle(annotated, (x_coord, y_coord), 5, color, -1)
                cv2.circle(annotated, (x_coord, y_coord), 7, (255, 255, 255), 2)
                
                # Add keypoint label
                label = self.KEYPOINT_NAMES[i]
                cv2.putText(annotated, label, (x_coord + 10, y_coord - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def get_keypoint_by_name(self, keypoints, name):
        """
        Get keypoint by name.

        Args:
            keypoints: Array of keypoints
            name: Name of keypoint (e.g., 'nose', 'left_shoulder')

        Returns:
            Keypoint array [y, x, confidence] or None
        """
        try:
            idx = self.KEYPOINT_NAMES.index(name)
            return keypoints[idx]
        except (ValueError, IndexError):
            return None
