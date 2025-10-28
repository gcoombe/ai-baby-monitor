#!/bin/bash

# AI Baby Monitor Setup Script
# This script helps set up the baby monitor on Raspberry Pi or Linux systems

set -e

echo "=========================================="
echo "AI Baby Monitor - Setup Script"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    if [[ $MODEL == *"Raspberry Pi"* ]]; then
        echo "Detected: $MODEL"
        IS_RPI=true
    fi
fi

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Install with: sudo apt-get install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Install system dependencies
echo ""
echo "Installing system dependencies..."
if [ "$IS_RPI" = true ]; then
    echo "Installing Raspberry Pi specific packages..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-opencv \
        libopencv-dev \
        portaudio19-dev \
        python3-pyaudio \
        python3-pip \
        python3-dev \
        python3-venv
else
    echo "Installing generic Linux packages..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-opencv \
        libopencv-dev \
        portaudio19-dev \
        python3-pip \
        python3-dev \
        python3-venv
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs

# Copy environment file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo ".env file created. Please edit it to add your API keys."
else
    echo ".env file already exists, skipping..."
fi

# Test camera
echo ""
echo "Testing camera..."
if [ "$IS_RPI" = true ]; then
    if raspistill --timeout 1 --output /tmp/test.jpg 2>/dev/null; then
        echo "Raspberry Pi camera is working!"
        rm /tmp/test.jpg
    else
        echo "Warning: Could not test Raspberry Pi camera"
        echo "You may need to enable it with: sudo raspi-config"
    fi
else
    if ls /dev/video* &> /dev/null; then
        echo "Camera devices found: $(ls /dev/video*)"
    else
        echo "Warning: No camera devices found"
    fi
fi

# Test audio
echo ""
echo "Testing audio..."
if command -v arecord &> /dev/null; then
    if arecord -l | grep -q "card"; then
        echo "Audio devices found!"
    else
        echo "Warning: No audio input devices found"
    fi
else
    echo "Warning: arecord not available, cannot test audio"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys: nano .env"
echo "2. Review config.yaml settings: nano config.yaml"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Run the monitor: python main.py"
echo "5. Access dashboard at: http://localhost:5000"
echo ""
echo "For Raspberry Pi, to run on startup:"
echo "  sudo cp baby-monitor.service /etc/systemd/system/"
echo "  sudo systemctl enable baby-monitor"
echo "  sudo systemctl start baby-monitor"
echo ""
