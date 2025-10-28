# ML Models for Video Detection - Implementation Guide

This guide covers ML models that would significantly enhance the baby monitor and demonstrate strong AI/ML skills for job applications.

## Recommended Model Stack

### 1. **MoveNet (Pose Estimation)** - PRIMARY RECOMMENDATION
**Why it's perfect for baby monitoring:**
- Detects body keypoints (head, shoulders, arms, legs)
- Determines baby's sleeping position (back, stomach, side)
- Safety feature: Alert if baby is face-down
- Lightweight enough for Raspberry Pi (MoveNet Lightning: ~6ms on Pi 4)
- Pre-trained and works out of the box

**What you'll learn:**
- Pose estimation algorithms
- TensorFlow Lite deployment
- Real-time inference optimization
- Safety-critical AI applications

**Implementation difficulty:** ⭐⭐⭐ (Medium)

### 2. **MobileNetV3 + Custom Classifier (Activity Recognition)** - BEST FOR PORTFOLIO
**Why it's perfect:**
- Transfer learning from pre-trained MobileNetV3
- Fine-tune on custom baby activity dataset
- Classifies: sleeping, awake-calm, awake-active, crying, playing
- Shows end-to-end ML pipeline (data collection → training → deployment)
- Great interview talking point

**What you'll learn:**
- Transfer learning
- Custom dataset creation
- Model training and optimization
- Quantization for edge devices
- Real-world model deployment

**Implementation difficulty:** ⭐⭐⭐⭐ (Medium-Hard, but most impressive)

### 3. **EfficientDet-Lite (Object Detection)** - SUPPLEMENTARY
**Why it's useful:**
- Confirms baby presence in crib
- Multi-object detection (baby, toys, blankets)
- Can detect if crib is empty
- Pre-trained models available

**What you'll learn:**
- Object detection architectures
- Bounding box predictions
- Multi-class detection

**Implementation difficulty:** ⭐⭐ (Easy, pre-trained available)

---

## Detailed Implementation: MoveNet Pose Estimation

### Why Start Here
1. **Immediate value**: Safety feature (detect unsafe positions)
2. **Pre-trained**: Works without training data
3. **Raspberry Pi optimized**: TFLite version available
4. **Impressive demo**: Visual keypoint overlay in dashboard

### Technical Specs
```
Model: MoveNet Lightning (Single Pose)
Size: 4.2 MB
Latency: 6-10ms on Raspberry Pi 4
Input: 192x192x3 RGB image
Output: 17 keypoints with confidence scores
```

### Implementation Code

```python
# monitors/pose_estimator.py
import tensorflow as tf
import numpy as np
import cv2

class BabyPoseEstimator:
    """Estimate baby's pose using MoveNet"""

    # 17 keypoints detected by MoveNet
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    def __init__(self, model_path='models/movenet_lightning.tflite'):
        """Load TFLite model"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_size = self.input_details[0]['shape'][1:3]  # [192, 192]

    def preprocess(self, frame):
        """Preprocess frame for model input"""
        # Resize to model input size
        img = cv2.resize(frame, tuple(self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 255] int8 or [0, 1] float32 depending on model
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        return input_data

    def predict(self, frame):
        """
        Run pose estimation on frame.

        Returns:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
        """
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0, 0]  # Shape: (17, 3)

        return keypoints_with_scores

    def detect_sleeping_position(self, keypoints, confidence_threshold=0.3):
        """
        Determine baby's sleeping position.

        Args:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
            confidence_threshold: Minimum confidence for keypoint

        Returns:
            position: 'back', 'stomach', 'side_left', 'side_right', 'unknown'
            confidence: Detection confidence
        """
        # Extract relevant keypoints
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Check if key points are visible
        visible_keypoints = [nose, left_shoulder, right_shoulder, left_hip, right_hip]
        avg_confidence = np.mean([kp[2] for kp in visible_keypoints])

        if avg_confidence < confidence_threshold:
            return 'unknown', avg_confidence

        # Calculate shoulder width (in image coordinates)
        shoulder_width = abs(left_shoulder[1] - right_shoulder[1])

        # Calculate hip width
        hip_width = abs(left_hip[1] - right_hip[1])

        # Calculate average width
        avg_width = (shoulder_width + hip_width) / 2

        # Determine position based on width and nose visibility
        if nose[2] < confidence_threshold:
            # Nose not visible - likely face down (UNSAFE!)
            return 'stomach', avg_confidence
        elif avg_width > 0.15:  # Threshold tuned for baby in crib
            # Wide shoulder/hip span - lying on back
            return 'back', avg_confidence
        else:
            # Narrow span - lying on side
            # Determine which side based on shoulder positions
            if left_shoulder[1] < right_shoulder[1]:
                return 'side_left', avg_confidence
            else:
                return 'side_right', avg_confidence

    def is_unsafe_position(self, position):
        """Check if position is unsafe (face down)"""
        return position == 'stomach'

    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.3):
        """
        Draw keypoints on frame for visualization.

        Args:
            frame: Original frame
            keypoints: Detected keypoints
            confidence_threshold: Minimum confidence to draw

        Returns:
            annotated_frame: Frame with keypoints drawn
        """
        height, width = frame.shape[:2]
        annotated = frame.copy()

        # Draw connections between keypoints
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

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
                cv2.circle(annotated, (x_coord, y_coord), 4, (0, 0, 255), -1)

        return annotated
```

### Integration with Video Monitor

```python
# Add to monitors/video_monitor.py

from monitors.pose_estimator import BabyPoseEstimator

class VideoMonitor:
    def __init__(self, config, db_manager, notification_manager):
        # ... existing code ...

        # Initialize pose estimator
        try:
            self.pose_estimator = BabyPoseEstimator()
            self.pose_enabled = True
            logger.info("Pose estimation enabled")
        except Exception as e:
            logger.warning(f"Pose estimation not available: {e}")
            self.pose_enabled = False

        self.current_position = 'unknown'
        self.unsafe_position_start = None

    def _monitor_loop(self):
        while self.running:
            ret, frame = self.camera.read()

            # ... existing motion detection code ...

            # Add pose estimation
            if self.pose_enabled and self.frame_count % 30 == 0:  # Every 30 frames
                keypoints = self.pose_estimator.predict(frame)
                position, confidence = self.pose_estimator.detect_sleeping_position(keypoints)

                self._handle_position_change(position, confidence)

                # Optionally draw keypoints on frame for dashboard
                if self.config.get('debug_mode', False):
                    frame = self.pose_estimator.draw_keypoints(frame, keypoints)

            self.frame_buffer.append(frame)

    def _handle_position_change(self, position, confidence):
        """Handle changes in baby's sleeping position"""
        if position != self.current_position:
            logger.info(f"Position changed: {self.current_position} → {position}")
            self.db.log_event(
                'position_change',
                confidence=confidence,
                details=f"Position: {position}"
            )
            self.current_position = position

            # Check for unsafe position
            if self.pose_estimator.is_unsafe_position(position):
                if not self.unsafe_position_start:
                    self.unsafe_position_start = datetime.now()
                else:
                    # If unsafe for more than 30 seconds, send urgent alert
                    duration = (datetime.now() - self.unsafe_position_start).total_seconds()
                    if duration > 30:
                        self.notifier.notify_unsafe_position(position, duration)
                        self.unsafe_position_start = None  # Reset to avoid spam
            else:
                self.unsafe_position_start = None
```

### Download and Setup

```bash
# Create models directory
mkdir -p models

# Download MoveNet Lightning model
cd models
wget https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite -O movenet_lightning.tflite

# Or use the Python script:
python download_models.py
```

```python
# download_models.py
import tensorflow_hub as hub
import tensorflow as tf

# Download and convert MoveNet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('models/movenet_lightning.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model downloaded and converted successfully!")
```

---

## Alternative Approach: Custom Activity Classifier

### Why This Is Better for Job Interviews
Building a custom model from scratch demonstrates:
1. Data collection and labeling
2. Model architecture design
3. Training and optimization
4. Deployment pipeline
5. End-to-end ML skills

### Dataset Creation

```python
# tools/collect_training_data.py
"""
Tool to collect training data for activity classification.
Run this while baby monitor is running to capture labeled frames.
"""

import cv2
import os
from datetime import datetime

class DataCollector:
    """Collect labeled frames for training"""

    ACTIVITIES = ['sleeping', 'awake_calm', 'awake_active', 'crying']

    def __init__(self, save_dir='training_data'):
        self.save_dir = save_dir
        for activity in self.ACTIVITIES:
            os.makedirs(f"{save_dir}/{activity}", exist_ok=True)

        self.camera = cv2.VideoCapture(0)

    def collect(self):
        """Interactive data collection"""
        print("Data Collection Tool")
        print("Keys: 1=sleeping, 2=awake_calm, 3=awake_active, 4=crying, q=quit")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue

            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                activity_idx = int(chr(key)) - 1
                activity = self.ACTIVITIES[activity_idx]

                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.save_dir}/{activity}/{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    collector.collect()
```

### Model Architecture

```python
# models/activity_classifier.py
"""
Custom CNN for baby activity classification using transfer learning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_activity_classifier(num_classes=4, input_shape=(224, 224, 3)):
    """
    Create activity classifier using MobileNetV3 backbone.

    Args:
        num_classes: Number of activity classes
        input_shape: Input image shape

    Returns:
        Keras model
    """
    # Load pre-trained MobileNetV3 (without top layers)
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model initially
    base_model.trainable = False

    # Add custom classification head
    inputs = keras.Input(shape=input_shape)

    # Preprocessing
    x = keras.applications.mobilenet_v3.preprocess_input(inputs)

    # Base model
    x = base_model(x, training=False)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout for regularization
    x = layers.Dropout(0.3)(x)

    # Dense layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model

def fine_tune_model(model, num_layers_to_unfreeze=20):
    """
    Unfreeze top layers of base model for fine-tuning.

    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze
    """
    base_model = model.layers[2]  # MobileNetV3 base
    base_model.trainable = True

    # Freeze all layers except the top ones
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    return model
```

### Training Script

```python
# train_activity_classifier.py
"""
Train the activity classification model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

from models.activity_classifier import create_activity_classifier, fine_tune_model

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 20
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001

def load_dataset(data_dir='training_data'):
    """Load and prepare dataset"""

    # Create dataset from directory
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Data augmentation for training
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def train_model():
    """Train the activity classifier"""

    print("Loading dataset...")
    train_ds, val_ds = load_dataset()

    print("Creating model...")
    model = create_activity_classifier(num_classes=4)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE_INITIAL),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]

    # Initial training (frozen base)
    print("\n=== Phase 1: Training with frozen base ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks
    )

    # Fine-tuning (unfreeze top layers)
    print("\n=== Phase 2: Fine-tuning ===")
    model = fine_tune_model(model)

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE_FINE_TUNE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks
    )

    # Convert to TFLite for Raspberry Pi
    print("\n=== Converting to TensorFlow Lite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('models/activity_classifier.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Training complete!")
    print(f"Model saved to: models/activity_classifier.tflite")

    # Evaluate
    test_loss, test_acc = model.evaluate(val_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model, history1, history2

if __name__ == '__main__':
    train_model()
```

---

## Performance Comparison

| Model | Latency (Pi 4) | Accuracy | Model Size | Difficulty | Portfolio Value |
|-------|---------------|----------|------------|------------|----------------|
| MoveNet Lightning | 6-10ms | N/A (pose) | 4.2 MB | Medium | ⭐⭐⭐⭐ |
| Custom Activity CNN | 20-30ms | 85-95%* | 3-5 MB | Hard | ⭐⭐⭐⭐⭐ |
| EfficientDet-Lite0 | 50-70ms | 90%+ | 4.4 MB | Easy | ⭐⭐⭐ |
| YOLOv5-nano | 30-40ms | 88%+ | 3.9 MB | Medium | ⭐⭐⭐⭐ |

*Depends on training data quality and quantity

---

## Recommended Implementation Order

### Week 1-2: MoveNet Integration
- Download and test model
- Integrate with video monitor
- Add position detection logic
- Update dashboard to show position
- **Result**: Working pose estimation system

### Week 3-4: Data Collection
- Run monitor and collect labeled frames
- Aim for 500-1000 images per class
- Can use existing videos/photos of your baby
- Data augmentation can help with limited data

### Week 5-6: Training Custom Model
- Set up training pipeline
- Train activity classifier
- Optimize and quantize for Pi
- Evaluate performance

### Week 7: Integration & Polish
- Integrate both models
- Update dashboard with visualizations
- Add model performance metrics
- Write documentation

---

## Interview Talking Points

### Technical Depth
1. **Model Selection**: "I chose MoveNet for pose estimation because it's optimized for edge devices, provides real-time inference on Raspberry Pi, and the keypoint detection allows me to determine baby's sleeping position for safety monitoring."

2. **Custom Training**: "For activity classification, I used transfer learning with MobileNetV3 as the backbone, collecting my own dataset and fine-tuning the model. This demonstrates end-to-end ML pipeline skills from data collection to deployment."

3. **Edge Optimization**: "I used TensorFlow Lite with quantization to reduce model size and inference time, crucial for running on Raspberry Pi with limited resources."

4. **Real-world Constraints**: "Had to balance accuracy with latency - running inference every 30 frames instead of every frame to maintain system responsiveness while still catching important events."

### Business Value
- **Safety**: Detects unsafe sleeping positions
- **Insights**: Tracks activity patterns over time
- **Automation**: Reduces false positives vs. simple motion detection
- **Scalability**: Model can be retrained as baby grows

---

## Additional Resources

### Pre-trained Models
- **TensorFlow Hub**: https://tfhub.dev
- **TF Lite Model Zoo**: https://www.tensorflow.org/lite/models
- **MediaPipe**: https://google.github.io/mediapipe/

### Training Data
- **Baby Activity Datasets** (if available publicly)
- **Create your own**: Best option for this project
- **Synthetic data**: Can augment real data

### Optimization Techniques
- Quantization (INT8, FP16)
- Pruning
- Knowledge distillation
- Model architecture search (NAS)

---

## Cost-Benefit Analysis

### MoveNet (Recommended Start)
**Pros:**
- Pre-trained, works immediately
- Safety-critical feature
- Fast inference
- No training data needed

**Cons:**
- Limited to pose, not activity
- Can't classify "crying" vs "awake"

### Custom Classifier (Best for Portfolio)
**Pros:**
- Shows full ML pipeline
- Custom to your use case
- Better for interviews
- More accurate activity detection

**Cons:**
- Requires training data collection
- More development time
- Need ML training experience

### Recommendation
**Start with MoveNet**, integrate it fully, then **add custom classifier** as enhancement. This gives you:
1. Working ML feature quickly
2. Safety functionality
3. Foundation for custom model
4. Progressive complexity (good for learning)

Both models combined create an impressive, production-quality ML system perfect for job applications in the AI space.
