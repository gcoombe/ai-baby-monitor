# Video Storage Feature Guide

The AI Baby Monitor now includes optional video recording capabilities to store footage for later review. This guide covers configuration, usage, and API endpoints for managing recordings.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Recording Modes](#recording-modes)
- [File Management](#file-management)
- [Web API Endpoints](#web-api-endpoints)
- [Storage Requirements](#storage-requirements)
- [Best Practices](#best-practices)

## Overview

The video storage feature allows you to record and store video footage from your baby monitor with the following capabilities:

- **Multiple Recording Modes**: Continuous, motion-triggered, or event-based recording
- **Automatic Retention Policies**: Automatically delete old recordings after a configurable period
- **Quality Settings**: Choose between low, medium, and high quality recordings
- **Timestamp Overlay**: Optional timestamps on recorded videos
- **Pre/Post Motion Buffering**: Capture video before and after motion events
- **Web API**: Manage, download, and stream recordings via REST API

## Configuration

Video storage is configured in `config.yaml` under the `video_storage` section:

```yaml
video_storage:
  enabled: false  # Set to true to enable video recording
  storage_path: "recordings"  # Directory to store recorded videos
  recording_mode: "motion"  # Options: continuous, motion, event
  segment_duration: 300  # Seconds per video file (5 minutes)
  retention_days: 7  # Days to keep recordings (0 = keep forever)
  quality: "medium"  # Options: low, medium, high
  include_timestamp: true  # Add timestamp overlay to videos
  pre_motion_buffer: 5  # Seconds of video before motion event
  post_motion_duration: 10  # Seconds to record after motion stops
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable or disable video recording |
| `storage_path` | string | `"recordings"` | Directory path for storing videos (relative or absolute) |
| `recording_mode` | string | `"motion"` | Recording trigger mode (see below) |
| `segment_duration` | integer | `300` | Maximum duration per video file in seconds |
| `retention_days` | integer | `7` | Days to keep recordings (0 = keep forever) |
| `quality` | string | `"medium"` | Video quality level |
| `include_timestamp` | boolean | `true` | Overlay timestamp on recorded videos |
| `pre_motion_buffer` | integer | `5` | Seconds of pre-motion footage to include |
| `post_motion_duration` | integer | `10` | Seconds to record after motion stops |

## Recording Modes

The system supports three recording modes:

### 1. Continuous Mode (`continuous`)

Records video continuously, creating new segment files every `segment_duration` seconds.

**Use Case**: When you want complete 24/7 coverage of the baby monitor.

**Pros**:
- Never miss any activity
- Complete timeline coverage

**Cons**:
- Highest storage requirements
- Generates large amounts of video data

**Configuration**:
```yaml
recording_mode: "continuous"
segment_duration: 300  # Creates a new 5-minute file every 5 minutes
```

### 2. Motion-Triggered Mode (`motion`)

Records only when motion is detected, with configurable pre and post-motion buffering.

**Use Case**: Save storage space while capturing all significant activity.

**Pros**:
- Efficient storage usage
- Captures all motion events
- Includes context before/after motion

**Cons**:
- May miss very subtle movements below motion threshold
- Gaps between motion events

**Configuration**:
```yaml
recording_mode: "motion"
pre_motion_buffer: 5  # 5 seconds before motion detected
post_motion_duration: 10  # 10 seconds after motion stops
```

**How it works**:
1. System maintains a rolling buffer of the last `pre_motion_buffer` seconds
2. When motion is detected, starts recording and writes buffered frames
3. Continues recording while motion is detected
4. After motion stops, records for `post_motion_duration` seconds
5. Stops recording and creates a new file for the next motion event

### 3. Event-Based Mode (`event`)

Records during significant events: when baby is awake or during significant motion.

**Use Case**: Capture important moments without recording every small movement.

**Pros**:
- Captures only meaningful events
- Lower storage than continuous mode
- Focuses on significant activity

**Cons**:
- May miss some activity
- Depends on accuracy of event detection

**Configuration**:
```yaml
recording_mode: "event"
segment_duration: 300  # Max 5 minutes per segment
```

**Triggers**:
- Baby state changes to awake
- Significant motion detected (motion area > 2x threshold)
- Records in segments of `segment_duration` seconds while events continue

## File Management

### Automatic Cleanup

The system automatically cleans up old recordings based on the `retention_days` setting:

- Cleanup runs every 6 hours automatically
- Deletes recordings older than `retention_days`
- Set `retention_days: 0` to disable automatic deletion and keep recordings forever

### Manual Cleanup

You can trigger manual cleanup via the API:

```bash
curl -X POST http://localhost:5000/api/recordings/cleanup
```

Response:
```json
{
  "files_deleted": 5,
  "bytes_freed": 1048576000,
  "mb_freed": 1000.0
}
```

### File Naming Convention

Recordings are saved with timestamps:

```
recording_YYYYMMDD_HHMMSS.mp4
```

Example: `recording_20250128_143052.mp4` (recorded on Jan 28, 2025 at 2:30:52 PM)

## Web API Endpoints

The system provides REST API endpoints for managing recordings:

### List Recordings

```
GET /api/recordings?limit=10&sort_by=date&ascending=false
```

**Query Parameters**:
- `limit` (optional): Maximum number of recordings to return
- `sort_by` (optional): Sort criterion (`date`, `size`, `name`)
- `ascending` (optional): Sort order (`true` or `false`)

**Response**:
```json
[
  {
    "filename": "recording_20250128_143052.mp4",
    "path": "/path/to/recordings/recording_20250128_143052.mp4",
    "size": 10485760,
    "created": "2025-01-28T14:30:52",
    "modified": "2025-01-28T14:35:52",
    "age_days": 0
  }
]
```

### Get Storage Statistics

```
GET /api/recordings/stats
```

**Response**:
```json
{
  "total_files": 42,
  "total_size_bytes": 1048576000,
  "total_size_mb": 1000.0,
  "total_size_gb": 0.98,
  "oldest_recording": "2025-01-21T14:30:52",
  "newest_recording": "2025-01-28T14:30:52",
  "storage_path": "/path/to/recordings"
}
```

### Download Recording

```
GET /api/recordings/{filename}/download
```

Downloads the recording file to the client's device.

**Example**:
```bash
curl -O http://localhost:5000/api/recordings/recording_20250128_143052.mp4/download
```

### Stream Recording

```
GET /api/recordings/{filename}/stream
```

Stream the recording for playback in a browser or media player.

**Example**:
```html
<video controls>
  <source src="http://localhost:5000/api/recordings/recording_20250128_143052.mp4/stream" type="video/mp4">
</video>
```

### Delete Recording

```
DELETE /api/recordings/{filename}
```

Delete a specific recording file.

**Example**:
```bash
curl -X DELETE http://localhost:5000/api/recordings/recording_20250128_143052.mp4
```

**Response**:
```json
{
  "message": "Recording recording_20250128_143052.mp4 deleted successfully"
}
```

### Manual Cleanup

```
POST /api/recordings/cleanup
```

Manually trigger cleanup of old recordings based on retention policy.

## Storage Requirements

### Storage Calculations

Storage requirements depend on your recording mode and quality settings:

**Approximate bitrates**:
- Low quality: ~500 Kbps (3.75 MB/minute, 225 MB/hour)
- Medium quality: ~1 Mbps (7.5 MB/minute, 450 MB/hour)
- High quality: ~2 Mbps (15 MB/minute, 900 MB/hour)

**Example storage requirements** (640x480 @ 10fps, medium quality):

| Recording Mode | Daily Storage | Weekly Storage |
|---------------|---------------|----------------|
| Continuous | ~10.8 GB | ~75.6 GB |
| Motion (50% active) | ~5.4 GB | ~37.8 GB |
| Event (25% active) | ~2.7 GB | ~18.9 GB |

### Recommendations

- **Raspberry Pi users**: Use an external USB drive or NAS for storage
- **Set retention_days**: Automatically manage storage by setting a retention period
- **Monitor storage**: Check `/api/recordings/stats` regularly
- **Use motion mode**: Best balance between coverage and storage efficiency

## Best Practices

### 1. Start with Motion Mode

Motion-triggered recording provides the best balance:
```yaml
recording_mode: "motion"
pre_motion_buffer: 5
post_motion_duration: 10
```

### 2. Set Appropriate Retention

Keep recordings for a reasonable period:
```yaml
retention_days: 7  # One week of recordings
```

### 3. Choose Medium Quality

Medium quality provides good video quality with reasonable file sizes:
```yaml
quality: "medium"
```

### 4. Enable Timestamps

Timestamps help identify when events occurred:
```yaml
include_timestamp: true
```

### 5. Monitor Storage

Regularly check storage statistics via the API to ensure you have sufficient space.

### 6. Configure Motion Detection

Adjust motion detection settings to reduce false triggers:
```yaml
detection:
  video:
    motion_threshold: 25
    motion_min_area: 500
```

### 7. Test Your Configuration

Before relying on recordings, test your setup:
1. Enable recording with a short retention period
2. Test each recording mode
3. Verify files are being created correctly
4. Check that cleanup works as expected

## Security Considerations

### File Access

- Recordings are stored locally and not uploaded to cloud services
- Access recordings only via the authenticated web API
- Ensure proper file system permissions on the `storage_path` directory

### Network Security

- Use HTTPS if accessing the web interface remotely
- Consider VPN access for remote monitoring
- Implement authentication for production deployments

### Privacy

- Be aware of privacy laws regarding video recording in your jurisdiction
- Secure your recordings with appropriate file system permissions
- Use retention policies to automatically delete old footage

## Troubleshooting

### Recordings Not Saving

1. Check that `enabled: true` in config.yaml
2. Verify the `storage_path` directory exists and is writable
3. Check logs for error messages
4. Ensure sufficient disk space

### Storage Filling Up

1. Reduce `retention_days` for more aggressive cleanup
2. Switch from continuous to motion mode
3. Lower quality setting
4. Increase storage capacity

### Poor Video Quality

1. Increase quality setting to "high"
2. Improve lighting conditions
3. Adjust camera resolution in config
4. Check network bandwidth (if streaming)

### Motion Mode Not Triggering

1. Adjust `motion_threshold` (lower = more sensitive)
2. Reduce `motion_min_area` to capture smaller movements
3. Check camera positioning and lighting
4. Review motion detection logs

## Example Use Cases

### Scenario 1: 24/7 Recording with Weekly Retention

```yaml
video_storage:
  enabled: true
  storage_path: "/mnt/external_drive/recordings"
  recording_mode: "continuous"
  segment_duration: 300
  retention_days: 7
  quality: "medium"
  include_timestamp: true
```

### Scenario 2: Motion-Triggered with Long-Term Storage

```yaml
video_storage:
  enabled: true
  storage_path: "recordings"
  recording_mode: "motion"
  segment_duration: 300
  retention_days: 30
  quality: "medium"
  include_timestamp: true
  pre_motion_buffer: 10
  post_motion_duration: 15
```

### Scenario 3: Event-Based for Important Moments Only

```yaml
video_storage:
  enabled: true
  storage_path: "recordings"
  recording_mode: "event"
  segment_duration: 600
  retention_days: 0  # Keep forever
  quality: "high"
  include_timestamp: true
```

## Getting Started

1. **Update configuration**:
   ```bash
   nano config.yaml
   ```

2. **Enable video storage**:
   ```yaml
   video_storage:
     enabled: true
     recording_mode: "motion"
     retention_days: 7
   ```

3. **Restart the baby monitor**:
   ```bash
   python main.py
   ```

4. **Verify recordings are being created**:
   ```bash
   ls -lh recordings/
   ```

5. **Access via web API**:
   ```bash
   curl http://localhost:5000/api/recordings/stats
   ```

## Support

For issues or questions:
- Check the logs at `logs/baby_monitor.log`
- Review configuration settings in `config.yaml`
- Ensure all dependencies are installed
- Check disk space and permissions

---

**Note**: Video storage is disabled by default to preserve privacy and storage space. Enable it only if you need recorded footage for review.
