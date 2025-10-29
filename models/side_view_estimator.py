"""
Side view pose estimator for baby monitoring.

Optimized for cameras positioned to the side of the baby.
"""
import numpy as np
from models.pose_estimator import BabyPoseEstimator


class SideViewBabyPoseEstimator(BabyPoseEstimator):
    """
    Pose estimator for camera positioned to the side of the baby.
    
    This configuration is best for detecting:
    - Back sleeping (baby facing camera, torso parallel to ground)
    - Stomach sleeping (baby facing away, torso parallel to ground)
    - Side sleeping (one shoulder higher, body vertical orientation)
    """
    
    def detect_sleeping_position(self, keypoints, confidence_threshold=0.3):
        """
        Determine baby's sleeping position from a side view.
        
        In side view:
        - Back: Face visible, both shoulders at similar height, body horizontal
        - Stomach: Face not visible, shoulders at similar height, body horizontal
        - Side: One shoulder significantly higher than the other, body more vertical
        - Sitting: Face and shoulders vertically aligned, head significantly above shoulders
        
        NOTE: Optimized for babies in sleep sacks - focuses on upper body (shoulders, 
        head, arms) rather than hips/legs which may be obscured.
        
        Args:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
            confidence_threshold: Minimum confidence for keypoint to be considered valid

        Returns:
            tuple: (position, confidence)
                position: 'back', 'stomach', 'side_left', 'side_right', 'sitting', 'unknown'
                confidence: Average confidence of detection
        """
        # Extract key upper body parts (visible outside sleep sack)
        nose = keypoints[self.NOSE]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]

        # Check if key upper body points are visible enough
        # Focus on parts that are outside the sleep sack
        key_parts = [nose, left_shoulder, right_shoulder]
        confidences = [kp[2] for kp in key_parts if kp[2] > 0.1]  # Filter out very low confidence
        
        if len(confidences) < 2:
            # Need at least 2 key upper body points
            return 'unknown', 0.0
            
        avg_confidence = np.mean(confidences)

        if avg_confidence < confidence_threshold:
            return 'unknown', avg_confidence

        # In side view, analyze vertical positions (y-coordinates)
        # Calculate shoulder height difference
        shoulder_height_diff = abs(left_shoulder[0] - right_shoulder[0])
        
        # Calculate torso orientation using shoulders and elbows (if available)
        # Average shoulder position
        shoulder_y = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_x = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Use elbows as lower body reference if available, otherwise use head-to-shoulder
        elbow_conf_avg = (left_elbow[2] + right_elbow[2]) / 2
        if elbow_conf_avg > confidence_threshold:
            # Use elbows as lower reference point
            reference_y = (left_elbow[0] + right_elbow[0]) / 2
            reference_x = (left_elbow[1] + right_elbow[1]) / 2
        else:
            # Use head (nose) as reference point
            reference_y = nose[0]
            reference_x = nose[1]
        
        # Calculate body orientation (vertical vs horizontal)
        body_height = abs(shoulder_y - reference_y)
        body_width = abs(shoulder_x - reference_x)
        
        # Ratio > 1 means body is more vertical (side position)
        # Ratio < 1 means body is more horizontal (back or stomach)
        if body_width > 0.01:  # Avoid division by zero
            body_ratio = body_height / body_width
        else:
            body_ratio = body_height / 0.01

        nose_visible = nose[2] > confidence_threshold

        # Check for sitting position first (head and shoulders vertically aligned)
        # Calculate horizontal alignment between head and shoulders
        horizontal_alignment = abs(shoulder_x - reference_x)
        
        # When sitting, head should be above shoulders (nose y < shoulder y)
        # and horizontally aligned (small x difference)
        head_above_shoulders = nose[0] < shoulder_y
        
        if nose_visible and shoulder_height_diff < 0.10 and horizontal_alignment < 0.15:
            # Face visible, shoulders aligned, and body parts vertically aligned
            if body_ratio > 1.5 and head_above_shoulders:
                # Body is vertical with head above - sitting up!
                return 'sitting', avg_confidence

        # Determine position based on shoulder alignment and body orientation
        if shoulder_height_diff < 0.10:
            # Shoulders at similar heights - lying flat (back or stomach)
            if body_ratio < 1.5:
                # Body is horizontal
                if nose_visible:
                    # Face visible - lying on back (SAFE)
                    return 'back', avg_confidence
                else:
                    # Face not visible - likely face down (UNSAFE!)
                    return 'stomach', avg_confidence
            else:
                # Body more vertical but shoulders aligned - unclear
                return 'unknown', avg_confidence
        else:
            # One shoulder higher than the other - lying on side
            # Determine which side based on which shoulder is higher
            if left_shoulder[0] < right_shoulder[0]:
                # Left shoulder higher (lower y value)
                return 'side_left', avg_confidence
            else:
                # Right shoulder higher
                return 'side_right', avg_confidence

