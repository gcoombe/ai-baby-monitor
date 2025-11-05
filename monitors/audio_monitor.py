"""
Audio monitoring module for detecting baby cries and sounds.
"""
import numpy as np
import logging
from threading import Thread, Event
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Try to import pyaudio, but handle if not available
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("PyAudio not available. Audio monitoring will be disabled.")
    AUDIO_AVAILABLE = False


class AudioMonitor:
    """Monitors audio for baby cries and significant sounds"""

    def __init__(self, config, db_manager, notification_manager):
        """
        Initialize audio monitor.

        Args:
            config: Configuration dictionary
            db_manager: DatabaseManager instance
            notification_manager: NotificationManager instance
        """
        self.config = config
        self.db = db_manager
        self.notifier = notification_manager

        if not AUDIO_AVAILABLE:
            logger.warning("Audio monitoring disabled: PyAudio not available")
            self.enabled = False
            return

        self.enabled = True

        # Audio settings
        self.device_index = config['audio']['device_index']
        self.sample_rate = config['audio']['sample_rate']
        self.chunk_size = config['audio']['chunk_size']
        self.channels = config['audio']['channels']

        # Detection settings
        self.cry_threshold = config['detection']['audio']['cry_threshold']
        self.noise_threshold = config['detection']['audio']['noise_threshold']

        # State tracking
        self.audio_interface = None
        self.stream = None
        self.running = False
        self.thread = None
        self.stop_event = Event()

        # Audio analysis
        self.last_cry_time = None
        self.last_noise_time = None
        self.noise_history = []
        self.history_size = 10

    def start(self):
        """Start audio monitoring in a separate thread"""
        if not self.enabled:
            logger.warning("Audio monitoring not enabled")
            return False

        if self.running:
            logger.warning("Audio monitor already running")
            return

        logger.info("Starting audio monitor...")

        try:
            self.audio_interface = pyaudio.PyAudio()

            # Open audio stream
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )

            self.running = True
            self.stop_event.clear()
            self.thread = Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

            logger.info("Audio monitor started")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio monitor: {e}")
            return False

    def stop(self):
        """Stop audio monitoring"""
        if not self.running:
            return

        logger.info("Stopping audio monitor...")
        self.running = False
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=5)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio_interface:
            self.audio_interface.terminate()

        logger.info("Audio monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running and not self.stop_event.is_set():
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Analyze audio
                self._analyze_audio(audio_array)



            except Exception as e:
                logger.error(f"Error in audio monitor loop: {e}")
                time.sleep(1)

    def _analyze_audio(self, audio_data):
        """
        Analyze audio data for crying and noise.

        Args:
            audio_data: NumPy array of audio samples
        """
        # Calculate RMS (Root Mean Square) for volume
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Track noise levels
        self.noise_history.append(rms)
        if len(self.noise_history) > self.history_size:
            self.noise_history.pop(0)

        # Check for significant noise
        if rms > self.noise_threshold:
            self.last_noise_time = datetime.now()

            # Analyze for crying patterns
            is_cry, cry_confidence = self._detect_cry(audio_data, rms)

            if is_cry:
                self._handle_cry_detection(cry_confidence)

    def _detect_cry(self, audio_data, rms):
        """
        Detect if audio contains baby crying.

        This is a simplified detection based on volume and frequency characteristics.
        For production use, consider using a trained ML model.

        Args:
            audio_data: NumPy array of audio samples
            rms: RMS volume level

        Returns:
            tuple: (is_cry, confidence)
        """
        # Simple heuristic: baby cries are typically loud and have certain frequency characteristics
        # This is a basic implementation - replace with ML model for better accuracy

        # Check volume threshold
        if rms < self.noise_threshold * 1.5:
            return False, 0.0

        # Calculate basic frequency characteristics using FFT
        try:
            fft = np.fft.rfft(audio_data)
            frequencies = np.fft.rfftfreq(len(audio_data), 1 / self.sample_rate)
            magnitudes = np.abs(fft)

            # Baby cries typically have fundamental frequency around 300-600 Hz
            cry_freq_range = (250, 650)
            cry_freq_mask = (frequencies >= cry_freq_range[0]) & (frequencies <= cry_freq_range[1])
            cry_freq_energy = np.sum(magnitudes[cry_freq_mask])

            # Calculate total energy
            total_energy = np.sum(magnitudes)

            if total_energy > 0:
                # Ratio of cry frequency energy to total energy
                cry_ratio = cry_freq_energy / total_energy

                # Simple threshold-based detection
                # In production, use a trained classifier
                if cry_ratio > 0.3:
                    # Check for sustained pattern
                    avg_noise = np.mean(self.noise_history)
                    if rms > avg_noise * 2:
                        confidence = min(cry_ratio * 1.5, 1.0)
                        return True, confidence

        except Exception as e:
            logger.error(f"Error in cry detection: {e}")

        return False, 0.0

    def _handle_cry_detection(self, confidence):
        """
        Handle cry detection.

        Args:
            confidence: Detection confidence
        """
        now = datetime.now()

        # Avoid duplicate notifications within short time window
        if self.last_cry_time and (now - self.last_cry_time).total_seconds() < 30:
            return

        self.last_cry_time = now

        if confidence > self.cry_threshold:
            logger.info(f"Cry detected with confidence: {confidence:.2%}")
            self.db.log_event('cry', confidence=confidence)
            self.notifier.notify_cry_detected(confidence)

    def get_status(self):
        """Get current monitoring status"""
        return {
            'running': self.running,
            'enabled': self.enabled,
            'last_cry': self.last_cry_time.isoformat() if self.last_cry_time else None,
            'last_noise': self.last_noise_time.isoformat() if self.last_noise_time else None,
            'current_noise_level': self.noise_history[-1] if self.noise_history else 0
        }
