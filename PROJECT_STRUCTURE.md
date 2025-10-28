# Project Structure

```
ai-baby-monitor/
│
├── main.py                          # Main application entry point
│   └── BabyMonitor class           # Orchestrates all components
│
├── config.yaml                      # Main configuration file
├── .env.example                     # Environment variables template
├── .env                            # Your actual secrets (git-ignored)
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── database/
│   ├── __init__.py
│   └── db_manager.py               # Database operations
│       ├── DatabaseManager         # Main DB class
│       ├── SleepSession model      # Sleep tracking table
│       └── Event model             # Event logging table
│
├── monitors/
│   ├── __init__.py
│   ├── video_monitor.py            # Video processing
│   │   └── VideoMonitor class
│   │       ├── Motion detection (OpenCV)
│   │       ├── Background subtraction
│   │       ├── Activity tracking
│   │       └── Sleep/wake state machine
│   │
│   └── audio_monitor.py            # Audio processing
│       └── AudioMonitor class
│           ├── Audio capture (PyAudio)
│           ├── Volume (RMS) calculation
│           ├── FFT frequency analysis
│           └── Cry detection heuristics
│
├── notifiers/
│   ├── __init__.py
│   └── notification_manager.py     # Notification system
│       └── NotificationManager class
│           ├── Console notifications
│           ├── Pushbullet integration
│           └── Custom webhook support
│
├── web/
│   ├── __init__.py
│   ├── app.py                      # Flask web server
│   │   └── API endpoints:
│   │       ├── /api/status         # Current system status
│   │       ├── /api/events         # Recent events
│   │       ├── /api/sleep-sessions # Sleep history
│   │       ├── /api/statistics     # Sleep statistics
│   │       └── /video-feed         # Live video stream
│   │
│   └── templates/
│       └── index.html              # Web dashboard UI
│
├── data/                           # Created at runtime
│   └── baby_monitor.db            # SQLite database (git-ignored)
│
├── logs/                           # Created at runtime
│   └── baby_monitor.log           # Application logs (git-ignored)
│
├── setup.sh                        # Automated setup script
├── baby-monitor.service            # Systemd service file
│
├── README.md                       # Complete documentation
├── QUICKSTART.md                   # Quick start guide
└── PROJECT_STRUCTURE.md            # This file
```

## Data Flow

```
Camera → VideoMonitor → Motion Detection → DatabaseManager → Event Log
                      ↓                    ↓
                  Activity Analysis    Sleep Session
                      ↓                    ↓
               Baby Awake/Asleep    NotificationManager → Pushbullet/Console
                      ↓
                 Web Dashboard ← Flask API


Microphone → AudioMonitor → Volume Analysis → DatabaseManager
                          ↓
                    FFT Analysis
                          ↓
                   Cry Detection → NotificationManager
```

## Key Components

### 1. Main Application (main.py)
- Loads configuration
- Initializes all components
- Manages component lifecycle
- Handles graceful shutdown

### 2. Video Monitor (monitors/video_monitor.py)
**Algorithms:**
- Background subtraction (MOG2)
- Morphological operations for noise reduction
- Contour detection for motion tracking
- State machine for sleep/wake transitions

**State Tracking:**
- Tracks last motion time
- Monitors activity/inactivity periods
- Implements hysteresis (30s awake, 5min asleep)

### 3. Audio Monitor (monitors/audio_monitor.py)
**Signal Processing:**
- RMS calculation for volume
- FFT for frequency analysis
- Cry frequency detection (250-650 Hz range)
- Noise baseline comparison

**Note:** Currently uses heuristics. Can be enhanced with ML model.

### 4. Database Manager (database/db_manager.py)
**Tables:**
- `sleep_sessions`: Tracks sleep periods with quality scores
- `events`: Logs all detections with timestamps and confidence

**Operations:**
- CRUD operations for sessions and events
- Statistics calculation
- Sleep quality scoring

### 5. Notification Manager (notifiers/notification_manager.py)
**Channels:**
- Console output (always available)
- Pushbullet (requires API key)
- Custom webhooks (requires endpoint)

**Event Filtering:**
- Configurable event type filtering
- Time-based deduplication
- Confidence thresholding

### 6. Web Dashboard (web/app.py + templates/index.html)
**Features:**
- Real-time status display
- Live video streaming (MJPEG)
- Event timeline
- Sleep statistics
- Auto-refresh (2s interval)

## Configuration System

### config.yaml
- Camera settings (device, resolution, rotation)
- Detection thresholds (motion, cry)
- Sleep tracking parameters
- Notification preferences
- Web server settings
- Logging configuration

### .env
- Sensitive credentials (API keys)
- Environment-specific overrides
- Git-ignored for security

## Threading Architecture

```
Main Thread
├── Signal handlers (SIGINT, SIGTERM)
└── Keeps process alive

VideoMonitor Thread (daemon)
├── Capture frames at configured FPS
├── Detect motion every check_interval
└── Update sleep state

AudioMonitor Thread (daemon)
├── Capture audio chunks
├── Analyze for crying
└── Store noise history

Flask Server Thread (daemon)
├── Serve web dashboard
├── Stream video feed
└── Provide REST API
```

## Database Schema

### sleep_sessions
```sql
CREATE TABLE sleep_sessions (
    id INTEGER PRIMARY KEY,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_minutes FLOAT,
    quality_score FLOAT,  -- 0-100, based on stir count
    stir_count INTEGER DEFAULT 0
);
```

### events
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- motion, cry, awake, asleep, stirring
    confidence FLOAT,  -- 0.0-1.0
    details VARCHAR(500),
    notification_sent BOOLEAN DEFAULT FALSE
);
```

## AI/ML Opportunities

Current implementation uses traditional computer vision and signal processing. Great opportunities to add ML:

1. **Cry Detection**
   - Train CNN on baby cry audio datasets
   - Deploy with TensorFlow Lite on Raspberry Pi
   - Files: `monitors/audio_monitor.py`

2. **Activity Classification**
   - Classify baby poses/activities from video
   - Use pose estimation models
   - Files: `monitors/video_monitor.py`

3. **Sleep Pattern Prediction**
   - Time series forecasting of sleep patterns
   - Anomaly detection for unusual patterns
   - Files: New module `models/sleep_predictor.py`

4. **Face Detection**
   - Confirm baby presence in crib
   - Alert if no face detected
   - Files: `monitors/video_monitor.py`

## Performance Considerations

**For Raspberry Pi 3/4:**
- Resolution: 640x480 or lower
- FPS: 5-10
- Check intervals: 2-3 seconds
- Disable GUI (headless mode)

**For Desktop/Laptop:**
- Can use higher resolutions
- Higher FPS for smoother video
- Lower check intervals for faster response

## Extending the Project

### Add New Notification Channel
1. Add method to `notifiers/notification_manager.py`
2. Update config schema in `config.yaml`
3. Document in README.md

### Add New Event Type
1. Define detection logic in monitor class
2. Call `db.log_event()` with new type
3. Add notification method if needed
4. Update web dashboard styling

### Add Hardware Integration
1. Create new module in root (e.g., `hardware/sensors.py`)
2. Initialize in `main.py`
3. Add configuration to `config.yaml`
4. Integrate with notification system

## Testing

**Manual Testing:**
1. Camera: Wave hand, check motion detection
2. Audio: Make noise, check volume levels
3. Sleep tracking: Wait for state transitions
4. Web: Check dashboard updates
5. Notifications: Verify alerts received

**For Production:**
- Add unit tests for each module
- Mock camera/microphone for CI/CD
- Integration tests for component interaction
- Performance tests on Raspberry Pi

## Security Notes

- `.env` file is git-ignored (contains secrets)
- Web dashboard has no authentication (local network only)
- Database has no sensitive information
- Webhook payloads should use HTTPS

**For Production:**
- Add authentication to web dashboard
- Use HTTPS with self-signed cert
- Implement rate limiting
- Add user management

## Useful Commands

```bash
# Start monitor
python main.py

# View logs
tail -f logs/baby_monitor.log

# Query database
sqlite3 data/baby_monitor.db "SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;"

# Check system service
sudo systemctl status baby-monitor

# Test camera
raspistill -o test.jpg

# Test audio
arecord -d 5 test.wav && aplay test.wav

# Monitor CPU usage
top -p $(pgrep -f main.py)
```

## Dependencies Explained

**Core:**
- `opencv-python`: Computer vision and video processing
- `numpy`: Numerical operations
- `pillow`: Image handling

**Audio:**
- `pyaudio`: Audio capture from microphone
- `scipy`: Signal processing utilities
- `librosa`: Audio analysis (optional, for advanced features)

**Web:**
- `flask`: Web framework for dashboard
- `flask-cors`: Cross-origin requests

**Database:**
- `sqlalchemy`: ORM for database operations

**Notifications:**
- `requests`: HTTP requests for webhooks
- `pushbullet.py`: Pushbullet API client

**ML/AI (Optional):**
- `tflite-runtime`: TensorFlow Lite for Raspberry Pi
- For custom models and edge inference

## File Sizes (Approximate)

```
main.py                    3 KB
database/db_manager.py     6 KB
monitors/video_monitor.py  10 KB
monitors/audio_monitor.py  7 KB
notifiers/notification_manager.py  5 KB
web/app.py                4 KB
web/templates/index.html  12 KB
config.yaml               2 KB
README.md                 15 KB

Total code: ~64 KB (excluding dependencies)
```

## Performance Metrics

**Typical Resource Usage (Raspberry Pi 4):**
- CPU: 15-30%
- RAM: 150-300 MB
- Disk: 10-50 MB (database grows over time)
- Network: Minimal (only when serving web dashboard)

**Latency:**
- Motion detection: ~100-200ms per frame
- Cry detection: ~500ms per audio chunk
- Notification: ~1-3s (depending on method)
- Web dashboard refresh: 2s

## Version History Ideas

**v1.0** (Current)
- Basic motion and cry detection
- Sleep tracking
- Web dashboard
- Multiple notification channels

**Future v2.0**
- ML-based cry detection
- Face detection
- Mobile app
- Temperature/humidity sensors
- Cloud backup of statistics

**Future v3.0**
- Multi-camera support
- Advanced analytics dashboard
- Sleep pattern predictions
- Smart home integration (lights, sounds)
- Parent app with two-way audio
