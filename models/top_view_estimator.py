"""
Top-down view pose estimator for baby monitoring.

Optimized for cameras positioned above the baby (looking down).
"""
import numpy as np
from models.pose_estimator import BabyPoseEstimator


class TopViewBabyPoseEstimator(BabyPoseEstimator):
    """
    Pose estimator for camera positioned above the baby (looking down).
    
    This configuration is best for detecting:
    - Back sleeping (safest - both shoulders and face visible)
    - Stomach sleeping (unsafe - face may not be visible)
    - Side sleeping (shoulders appear at different depths)
    """
    
    def detect_sleeping_position(self, keypoints, confidence_threshold=0.3):
        """
        Determine baby's sleeping position from a top-down view.
        
        In top-down view:
        - Back: Wide shoulder span, face clearly visible, arms may be spread
        - Stomach: Wide shoulder span, face not visible or occluded
        - Side: Narrower shoulder span, one shoulder more prominent than the other
        - Sitting: Very narrow profile, face and shoulders very close together
        
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

        # Calculate shoulder width (in normalized coordinates 0-1)
        shoulder_width = abs(left_shoulder[1] - right_shoulder[1])

        # Also consider elbow span if visible (may be more reliable than hips in sleep sack)
        elbow_conf_avg = (left_elbow[2] + right_elbow[2]) / 2
        if elbow_conf_avg > confidence_threshold:
            elbow_width = abs(left_elbow[1] - right_elbow[1])
            # Use average of shoulder and elbow width
            body_width = (shoulder_width + elbow_width) / 2
        else:
            # Just use shoulder width
            body_width = shoulder_width

        # Determine position based on visible keypoints and body width
        nose_visible = nose[2] > confidence_threshold
        
        # Calculate distance between nose and shoulder center
        shoulder_center_y = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
        nose_to_shoulder_dist = np.sqrt(
            (nose[0] - shoulder_center_y)**2 + 
            (nose[1] - shoulder_center_x)**2
        )

        # Check for sitting position first (very compact from top view)
        if nose_visible and body_width < 0.08 and nose_to_shoulder_dist < 0.10:
            # Very narrow profile with face and shoulders very close - sitting up!
            return 'sitting', avg_confidence

        # Adjusted thresholds for upper body only detection
        if body_width > 0.12:
            # Wide shoulder/elbow span - lying flat (back or stomach)
            if nose_visible:
                # Nose visible and wide body - lying on back (SAFE)
                return 'back', avg_confidence
            else:
                # Nose not visible and wide body - likely face down (UNSAFE!)
                return 'stomach', avg_confidence

        elif body_width > 0.04:
            # Moderate span - lying on side
            # Determine which side based on relative shoulder confidence and positions
            # When on side, the upper shoulder is typically more visible
            left_shoulder_conf = left_shoulder[2]
            right_shoulder_conf = right_shoulder[2]
            
            if left_shoulder_conf > right_shoulder_conf:
                # Left shoulder more visible - lying on right side (left shoulder up)
                return 'side_right', avg_confidence
            else:
                # Right shoulder more visible - lying on left side (right shoulder up)
                return 'side_left', avg_confidence

        else:
            # Very narrow or unclear
            return 'unknown', avg_confidence

