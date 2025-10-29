# Testing Baby Monitor Algorithm on Recorded Videos

This guide explains how to test your baby monitor's detection algorithm on pre-recorded video files using the `test_video.py` script.

## Quick Start

### Basic Usage

Test a video file with default settings:
```bash
python test_video.py path/to/your/video.mp4
```

### View Video While Processing

Watch the video with detection overlays in real-time:
```bash
python test_video.py video.mp4 --show-video
```

**Controls during playback:**
- Press `q` to quit
- Press `Space` to pause/resume

### Save Annotated Output

Process the video and save an annotated version:
```bash
python test_video.py video.mp4 --save-output annotated_video.mp4
```

### Generate Analysis Report

Save a detailed analysis report:
```bash
python test_video.py video.mp4 --report analysis_report.txt
```

This creates two files:
- `analysis_report.txt` - Human-readable text report
- `analysis_report.json` - Machine-readable JSON data

## Complete Example

Process a video with all features:
```bash
python test_video.py baby_nap_recording.mp4 \
  --show-video \
  --save-output analyzed_nap.mp4 \
  --report nap_analysis.txt \
  --rotation 180
```

## Configuration Options

### Video Rotation

If your video is oriented incorrectly:
```bash
python test_video.py video.mp4 --rotation 90    # Rotate 90° clockwise
python test_video.py video.mp4 --rotation 180   # Rotate 180°
python test_video.py video.mp4 --rotation 270   # Rotate 270° clockwise
```

### Motion Detection Sensitivity

Adjust motion detection thresholds:
```bash
# More sensitive (detects smaller movements)
python test_video.py video.mp4 --motion-threshold 15 --motion-min-area 300

# Less sensitive (only detects larger movements)
python test_video.py video.mp4 --motion-threshold 40 --motion-min-area 800
```

**Parameters:**
- `--motion-threshold`: Background subtraction sensitivity (default: 25, lower = more sensitive)
- `--motion-min-area`: Minimum pixel area to consider motion (default: 500)

### Logging Level

Control output verbosity:
```bash
python test_video.py video.mp4 --log-level DEBUG   # Detailed debug info
python test_video.py video.mp4 --log-level INFO    # Normal info (default)
python test_video.py video.mp4 --log-level WARNING # Only warnings and errors
```

## Features

### Motion Detection

The script analyzes each frame for motion using background subtraction:
- **Basic Motion**: Any movement detected
- **Significant Motion**: Larger movements indicating baby stirring or being awake

### Sleep/Wake State Tracking

Monitors sustained activity to determine if baby is:
- **Asleep**: No significant motion for 5 minutes (configurable)
- **Awake**: Continuous activity for 30 seconds (configurable)

The report shows all state transitions with timestamps.

### Pose Estimation (Optional)

If TensorFlow is installed and the MoveNet model is available:
- Detects baby's body keypoints (17 points)
- Identifies sleeping position: back, side, stomach, or unknown
- **Alerts on unsafe positions** (face-down on stomach)

To enable pose estimation:
```bash
# Install TensorFlow if not already installed
pip install tensorflow

# Download the MoveNet model
python download_models.py
```

### Visual Annotations

When using `--show-video` or `--save-output`, the video includes:
- **Green contours**: Areas with detected motion
- **Colored keypoints**: Body parts detected by pose estimation
  - Blue: Face (nose, eyes, ears)
  - Green: Upper body (shoulders, elbows, wrists)
  - Red: Lower body (hips, knees, ankles)
- **Text overlays**: Current time, sleep state, motion level, sleeping position
- **Warnings**: Red alert for unsafe sleeping positions

## Understanding the Report

### Video Information
- Duration, frame count, FPS, resolution
- Processing time

### Motion Detection
- Percentage of frames with motion
- Number of significant motion events
- Helps understand activity level throughout nap

### Sleep/Wake Analysis
- Lists all state transitions
- Shows when baby woke up or fell asleep
- Includes frame numbers and timestamps

### Sleeping Positions
- Distribution of detected positions
- Percentage of time in each position
- Only available if pose estimation is enabled

### Unsafe Position Alerts
- ⚠️ Lists any frames where baby appears face-down
- Critical safety information
- Includes timestamps for quick video review

## Example Report

```
======================================================================
BABY MONITOR VIDEO ANALYSIS REPORT
======================================================================

VIDEO INFORMATION:
  File: baby_nap.mp4
  Duration: 1800.00 seconds (30.00 minutes)
  Total Frames: 18000
  FPS: 10.00
  Resolution: 640x480
  Processing Time: 45.23 seconds

MOTION DETECTION:
  Frames with motion: 450 (2.5%)
  Frames with significant motion: 120 (0.7%)
  Total motion events: 15

SLEEP/WAKE ANALYSIS:
  Total transitions: 2
  Transitions:
    - Frame 5400 (0:09:00): AWAKE
    - Frame 7200 (0:12:00): ASLEEP

SLEEPING POSITIONS:
  back: 12000 frames (85.7%)
  side_left: 1800 frames (12.9%)
  side_right: 200 frames (1.4%)

SUMMARY:
  ✓ Minimal motion - baby appears calm
  ✓ No unsafe positions detected

======================================================================
```

## Tips for Testing

### 1. Start Simple
Begin with basic processing to verify everything works:
```bash
python test_video.py video.mp4
```

### 2. Review With Visualization
Use `--show-video` to understand what the algorithm detects:
```bash
python test_video.py video.mp4 --show-video
```

### 3. Adjust Sensitivity
If motion detection is too sensitive or not sensitive enough, adjust thresholds:
```bash
# Too many false positives? Reduce sensitivity:
python test_video.py video.mp4 --motion-threshold 35 --motion-min-area 700

# Missing real movements? Increase sensitivity:
python test_video.py video.mp4 --motion-threshold 15 --motion-min-area 300
```

### 4. Save Results for Comparison
Create annotated videos to compare algorithm versions:
```bash
python test_video.py video.mp4 \
  --save-output results_v1.mp4 \
  --report report_v1.txt

# After adjusting settings:
python test_video.py video.mp4 \
  --motion-threshold 20 \
  --save-output results_v2.mp4 \
  --report report_v2.txt
```

### 5. Check Pose Estimation Accuracy
If you see incorrect position detection:
- Ensure good lighting in your video
- Camera should have clear view of baby
- Baby should be fully visible in frame
- Consider the model's limitations with very small babies

## Video Format Support

The script supports common video formats:
- MP4 (recommended)
- AVI
- MOV
- MKV
- Any format supported by OpenCV

## Performance

Processing speed depends on:
- Video resolution (lower = faster)
- Frame rate (lower = faster)
- Pose estimation enabled/disabled (disabled = much faster)
- Your computer's CPU/GPU

**Typical processing speed:**
- Without pose estimation: 2-5x real-time
- With pose estimation: 0.5-2x real-time

Example: A 30-minute video might take 6-15 minutes to process with pose estimation, or 3-6 minutes without.

## Troubleshooting

### "Could not open video file"
- Verify the file path is correct
- Ensure the video file is not corrupted
- Try a different video format

### "TensorFlow not available"
Pose estimation requires TensorFlow:
```bash
pip install tensorflow
python download_models.py
```

### "Failed to load pose estimation model"
Download the required model:
```bash
python download_models.py
```

### Processing is too slow
- Disable pose estimation (it's optional)
- Use `--show-video` sparingly (rendering slows processing)
- Consider reducing video resolution before processing

### Out of memory errors
- Close other applications
- Process shorter video segments
- Reduce video resolution

## Integration with Real Monitoring

After testing on recorded videos:

1. **Tune your config.yaml** based on test results
2. **Adjust motion thresholds** in the configuration
3. **Verify sleep/wake timing** parameters work for your baby
4. **Run the main monitor** with confidence:
   ```bash
   python main.py
   ```

## Advanced: Batch Processing

Process multiple videos:
```bash
#!/bin/bash
for video in recordings/*.mp4; do
    echo "Processing $video..."
    python test_video.py "$video" \
      --save-output "analyzed/$(basename $video)" \
      --report "reports/$(basename $video .mp4).txt"
done
```

## Getting Help

For more options and help:
```bash
python test_video.py --help
```

For issues or questions about the baby monitor system, see:
- `README.md` - Main documentation
- `QUICKSTART.md` - Setup guide
- `ML_MODELS_GUIDE.md` - Pose estimation details


