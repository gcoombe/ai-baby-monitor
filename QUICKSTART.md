# Quick Start Guide

Get your AI Baby Monitor running in 5 minutes!

## Prerequisites

- Raspberry Pi with camera OR computer with webcam
- Python 3.8+
- Internet connection

## Installation

### Option 1: Automated Setup (Recommended for Raspberry Pi/Linux)

```bash
cd ai-baby-monitor
chmod +x setup.sh
./setup.sh
```

The script will:
- Install system dependencies
- Create virtual environment
- Install Python packages
- Set up directories
- Test camera and audio

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs

# Copy environment file
cp .env.example .env
```

## Configuration

### Basic (Required)

1. **Edit camera settings in config.yaml:**
```yaml
camera:
  device: 0  # Try 0, 1, or 2 if camera doesn't work
  resolution: [640, 480]  # Lower for Raspberry Pi
```

2. **Test camera:**
```bash
# On Raspberry Pi
raspistill -o test.jpg

# On other systems
ls /dev/video*  # Should show video devices
```

### Optional Notifications

1. **Get Pushbullet API key:**
   - Go to https://www.pushbullet.com
   - Sign up/login
   - Go to Settings → Account → Access Token
   - Copy the token

2. **Add to .env file:**
```bash
nano .env
# Add: PUSHBULLET_API_KEY=your_token_here
```

3. **Enable in config.yaml:**
```yaml
notifications:
  methods:
    - console
    - pushbullet  # Add this line
```

## Running

### Start the Monitor

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run the application
python main.py
```

You should see:
```
Starting AI Baby Monitor
Starting video monitor...
Video monitor started
Starting audio monitor...
AI Baby Monitor is running
```

### Access the Dashboard

Open your browser:
- **Local:** http://localhost:5000
- **Remote:** http://YOUR_PI_IP:5000

Find your Pi's IP:
```bash
hostname -I
```

## Testing

### Test Video Detection
1. Wave your hand in front of the camera
2. Check console for "motion detected" messages
3. View the web dashboard to see live feed

### Test Sleep Tracking
1. Let the monitor run for a few minutes with no motion
2. It should start a sleep session
3. Create motion to trigger "stirring" detection
4. Sustained motion will mark baby as "awake"

## Common Issues

### Camera not working
```bash
# Try different device numbers
camera:
  device: 0  # Try 0, 1, 2, etc.
```

### High CPU on Raspberry Pi
```yaml
camera:
  resolution: [320, 240]  # Use lower resolution
  fps: 5  # Reduce frame rate
```

### Can't access dashboard remotely
```bash
# Check firewall
sudo ufw allow 5000

# Or disable firewall temporarily
sudo ufw disable
```

## Next Steps

1. **Adjust sensitivity:**
   - Edit `config.yaml`
   - Change `motion_threshold` (lower = more sensitive)
   - Change `cry_threshold`

2. **Position camera:**
   - Mount camera above crib
   - Ensure good view of baby
   - Adjust `rotation` in config if needed

3. **Set up auto-start** (Raspberry Pi):
```bash
sudo cp baby-monitor.service /etc/systemd/system/
sudo systemctl enable baby-monitor
sudo systemctl start baby-monitor
```

4. **Monitor logs:**
```bash
# View application logs
tail -f logs/baby_monitor.log

# View system service logs
sudo journalctl -u baby-monitor -f
```

## What's Monitored

The system tracks:
- **Motion:** Any movement in the frame
- **Stirring:** Significant motion while asleep
- **Awake:** Sustained activity (30 seconds default)
- **Crying:** Audio patterns matching baby cries
- **Sleep Quality:** Based on stir frequency

## Sleep Session Logic

- **Falls Asleep:** After 5 minutes of no significant motion
- **Wakes Up:** After 30 seconds of sustained activity
- **Stirring:** Brief motion during sleep (doesn't wake)

Adjust these in `config.yaml`:
```yaml
sleep_tracking:
  awake_confirmation_time: 30  # seconds
  sleep_confirmation_time: 300  # seconds
```

## Getting Notifications

You'll receive alerts for:
- Baby waking up
- Baby stirring in sleep
- Crying detected
- Extended wake periods

Configure which events trigger notifications in `config.yaml`.

## Viewing Data

### Web Dashboard
- Real-time status
- Live video feed
- Sleep statistics
- Event history

### Database
```bash
sqlite3 data/baby_monitor.db
.tables
SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;
SELECT * FROM sleep_sessions ORDER BY start_time DESC;
```

## Need Help?

1. Check `logs/baby_monitor.log`
2. Enable debug mode in `config.yaml`:
```yaml
logging:
  level: DEBUG
```
3. Review README.md for detailed troubleshooting

## Project Ideas

Enhance this project by adding:
- Custom ML model for cry detection
- Face detection/recognition
- Sleep pattern predictions
- Mobile app
- Temperature monitoring
- Smart light integration

This project demonstrates real-world AI/ML skills:
- Computer vision (OpenCV)
- Signal processing
- Real-time systems
- Web development
- Database design
- Deployment on edge devices

Perfect for your portfolio!
