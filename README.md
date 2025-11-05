# AI Baby Monitor

An intelligent baby monitoring system that uses computer vision and audio processing to track your baby's sleep patterns, detect when they wake up or stir, and send real-time notifications. Designed to run on Raspberry Pi.

## Features

- **Video Monitoring**: Real-time motion detection and activity tracking using computer vision
- **Audio Monitoring**: Cry detection and sound analysis
- **Sleep Tracking**: Automatically tracks sleep sessions, duration, and quality
- **Smart Notifications**: Alerts you when:
  - Baby wakes up
  - Baby is stirring
  - Crying is detected
  - Extended wake periods
- **Web Dashboard**: Real-time monitoring interface with live video feed and statistics
- **Event Logging**: Complete history of all events and sleep sessions
- **Configurable**: Highly customizable detection thresholds and notification preferences


## Hardware Requirements

### Raspberry Pi Setup
- Raspberry Pi 3B+ or newer (Pi 5 recommended for better performance)
- Raspberry Pi Camera Module or USB webcam
- USB microphone (optional, for audio monitoring)
- MicroSD card (64GB+ recommended)
- Power supply

### Alternative Setup
The application can also run on:
- Desktop/laptop with webcam and microphone
- Any Linux-based system with camera access

## Software Requirements

- Python 3.8 or higher
- Raspberry Pi OS (or any Linux distribution)
- Internet connection (for notifications)

## Installation

### On Raspberry Pi

1. **Update your system**
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

2. **Install system dependencies**
```bash
# Install OpenCV dependencies
sudo apt-get install -y python3-opencv libopencv-dev

# Install audio dependencies
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Install other dependencies
sudo apt-get install -y python3-pip python3-dev
```

3. **Clone or download this project**
```bash
git clone git@github.com:gcoombe/ai-baby-monitor.git ai-baby-monitor
cd ai-baby-monitor
```

4. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

5. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

### On Desktop/Laptop (mac)

1. **Install Python 3.8+**

2. **Clone the project**
```bash
git clone <your-repo-url> ai-baby-monitor
cd ai-baby-monitor
```

3. **Create virtual environment**
```bash
python -m venv venv

source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Configuration

1. **Copy the example environment file**
```bash
cp .env.example .env
```

2. **Edit .env with your credentials** (optional)
```bash
nano .env
```

Add your Pushbullet API key if you want push notifications:
```
PUSHBULLET_API_KEY=your_api_key_here
```

Get your API key from: https://www.pushbullet.com/#settings/account

3. **Edit config.yaml** to customize settings
```bash
nano config.yaml
```

Key settings to adjust:
- `camera.device`: Camera index (0 for default, 1 for second camera)
- `camera.resolution`: Lower resolution for Raspberry Pi (e.g., [640, 480])
- `camera.rotation`: Rotate camera feed if needed (0, 90, 180, 270)
- `camera.view`: Camera perspective ("top" for overhead, "side" for side view)
- `detection.video.motion_threshold`: Lower = more sensitive
- `detection.audio.cry_threshold`: Confidence threshold for cry detection
- `notifications.methods`: Choose notification channels (console, pushbullet, webhook)

### Video Recording & Annotations

The system can record video with helpful visual overlays:

```yaml
video_storage:
  enabled: false  # Set to true to enable recording
  recording_mode: "motion"  # Options: continuous, motion, event
  include_timestamp: true  # Show timestamp on video
  include_annotations: true  # Show motion detection, state, and position overlays
```

**Recording Modes:**
- `continuous`: Records all the time in segments
- `motion`: Only records when motion is detected (saves space)
- `event`: Records when baby is awake or stirring

**Annotations** (when `include_annotations: true`):
- **Motion contours**: Green outlines around detected movement
- **Sleep state**: Shows "AWAKE" (red) or "ASLEEP" (green)
- **Motion level**: Displays motion intensity in pixels
- **Position detection**: Shows detected sleeping position and confidence (if pose estimation enabled)
- **Unsafe position alerts**: Red warning if baby is in an unsafe position
- **Timestamp**: Current date and time overlay

These annotations are identical to what you see when testing with `test_video.py`, making it easy to review recorded footage and understand what the monitor detected.

## Usage

### Basic Usage

1. **Activate virtual environment**
```bash
source venv/bin/activate  
```

2. **Run the monitor**
```bash
python main.py
```

3. **Access the web dashboard**
Open your browser and go to:
```
http://localhost:5000
```

Or from another device on the same network:
```
http://<raspberry-pi-ip>:5000
```

To find your Raspberry Pi IP:
```bash
hostname -I
```

### Testing with Recorded Videos

Before running the monitor live, you can test the algorithm on pre-recorded video files:

```bash
# Basic test - generates a text report
python test_video.py your_video.mp4 --report analysis.txt

# Watch the video with detection overlays
python test_video.py your_video.mp4 --show-video

# Save annotated video with all detections visualized
python test_video.py your_video.mp4 --save-output annotated.mp4 --report report.txt

```

The test script provides:
- **Motion detection analysis** - Shows when and how much movement was detected
- **Sleep/wake state tracking** - Identifies when baby appears awake or asleep
- **Position detection** (if TensorFlow installed) - Identifies sleeping positions and unsafe positions
- **Annotated video output** - Visual overlay showing all detections
- **Detailed reports** - Text and JSON reports with statistics and timeline

This is useful for:
- Testing the algorithm before live deployment
- Tuning detection sensitivity settings
- Reviewing past recordings
- Validating detection accuracy

See `TEST_VIDEO_GUIDE.md` for complete documentation.

### Running as a System Service (Raspberry Pi)

To run the monitor automatically on startup:

1. **Create a systemd service file**
```bash
sudo nano /etc/systemd/system/baby-monitor.service
```

2. **Add the following content** (adjust paths as needed):
```ini
[Unit]
Description=AI Baby Monitor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ai-baby-monitor
Environment="PATH=/home/pi/ai-baby-monitor/venv/bin"
ExecStart=/home/pi/ai-baby-monitor/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. **Enable and start the service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable baby-monitor
sudo systemctl start baby-monitor
```

4. **Check status**
```bash
sudo systemctl status baby-monitor
```

5. **View logs**
```bash
sudo journalctl -u baby-monitor -f
```

## Web Dashboard

The web dashboard provides:
- Live video feed from the camera
- Current monitoring status (running, baby awake/asleep)
- Current sleep session duration and stir count
- 7-day sleep statistics
- Recent events timeline with confidence scores
- Auto-refreshing every 2 seconds

## Notification Options

### Console Notifications
Prints alerts to the terminal/logs (enabled by default)

### Pushbullet Notifications
Sends push notifications to your phone and devices:
1. Create account at https://www.pushbullet.com
2. Get API key from account settings
3. Add to `.env` file
4. Enable in `config.yaml` under `notifications.methods`

### Custom Webhook
Send notifications to your own endpoint:
1. Set webhook URL in `.env` or `config.yaml`
2. Enable in `config.yaml` under `notifications.methods`

Webhook payload format:
```json
{
  "title": "Baby is Awake!",
  "message": "Your baby has woken up and is active.",
  "event_type": "baby_awake",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Camera Setup

### Raspberry Pi Camera Module
1. Enable camera interface:
```bash
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
```

2. Reboot:
```bash
sudo reboot
```

3. Test camera:
```bash
raspistill -o test.jpg
```

### USB Webcam
Should work automatically. If you have multiple cameras, adjust `camera.device` in `config.yaml`.

## Troubleshooting

### Camera not detected
```bash
# List available cameras
ls /dev/video*

# Try different device numbers in config.yaml (0, 1, 2, etc.)
```

### Audio not working
```bash
# List audio devices
arecord -l

# Test microphone
arecord -d 5 test.wav
aplay test.wav
```

### High CPU usage on Raspberry Pi
- Reduce camera resolution in `config.yaml` (e.g., [320, 240])
- Reduce FPS (e.g., 5-10 fps)
- Increase `check_interval` values

### Web dashboard not accessible
```bash
# Check if Flask is running
sudo netstat -tulpn | grep 5000

# Allow port through firewall (if applicable)
sudo ufw allow 5000
```

## Database

Sleep data and events are stored in SQLite database at `data/baby_monitor.db`.

To query the database directly:
```bash
sqlite3 data/baby_monitor.db
```

Example queries:
```sql
-- View recent events
SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;

-- View sleep sessions
SELECT * FROM sleep_sessions ORDER BY start_time DESC;

-- Calculate average sleep duration
SELECT AVG(duration_minutes) FROM sleep_sessions WHERE end_time IS NOT NULL;
```


## Performance Tips for Raspberry Pi

1. **Optimize for Pi 3/4**
```yaml
camera:
  resolution: [640, 480]  # or [320, 240] for Pi 3
  fps: 5-10

detection:
  video:
    check_interval: 3  # Check every 3 seconds
  audio:
    check_interval: 2
```

2. **Disable GUI** (headless mode)
```bash
sudo systemctl set-default multi-user.target
```

3. **Overclock** (optional, Pi 4)
```bash
sudo nano /boot/config.txt
# Add: over_voltage=2, arm_freq=1750
```


## License

MIT License - feel free to use this for personal or commercial projects.
