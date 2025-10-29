# Testing Quick Start

Quick reference for testing the baby monitor algorithm on recorded videos.

## One-Line Test

```bash
python test_video.py your_video.mp4
```

## Common Commands

### See it in action
```bash
python test_video.py video.mp4 --show-video
```
Controls: Press `q` to quit, `Space` to pause

### Save annotated video
```bash
python test_video.py video.mp4 --save-output results.mp4
```

### Generate report
```bash
python test_video.py video.mp4 --report analysis.txt
```

### Complete analysis
```bash
python test_video.py video.mp4 \
  --show-video \
  --save-output analyzed.mp4 \
  --report report.txt
```

### Use example script
```bash
./example_test.sh your_video.mp4
```
Creates timestamped directory with all outputs.

## Adjust Settings

### Rotate video
```bash
python test_video.py video.mp4 --rotation 180
```

### Adjust sensitivity
```bash
# More sensitive
python test_video.py video.mp4 --motion-threshold 15 --motion-min-area 300

# Less sensitive  
python test_video.py video.mp4 --motion-threshold 40 --motion-min-area 800
```

## What You Get

### Text Report
- Motion detection statistics
- Sleep/wake transitions with timestamps
- Sleeping position analysis (if TensorFlow installed)
- Unsafe position alerts
- Video processing stats

### JSON Data
Machine-readable version for further analysis

### Annotated Video
Visual overlay showing:
- Motion detection (green contours)
- Body keypoints (colored dots)
- Current state (awake/asleep)
- Sleeping position
- Unsafe position warnings

## Typical Workflow

1. **Record a test video** of your baby's nap
2. **Run basic test** to see if it works:
   ```bash
   python test_video.py video.mp4
   ```
3. **Watch with visualization** to verify detection:
   ```bash
   python test_video.py video.mp4 --show-video
   ```
4. **Adjust sensitivity** if needed and retest
5. **Update config.yaml** with tested values
6. **Run live monitor** with confidence:
   ```bash
   python main.py
   ```

## Troubleshooting

**TensorFlow not available?**
```bash
pip install tensorflow
python download_models.py
```

**Video won't open?**
- Check file path
- Try different format (MP4 recommended)

**Processing too slow?**
- Skip `--show-video` while processing
- Pose estimation adds overhead (but provides position detection)

## Full Documentation

See `TEST_VIDEO_GUIDE.md` for complete documentation.

## Get Help

```bash
python test_video.py --help
```


