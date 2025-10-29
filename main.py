"""
AI Baby Monitor - Main Application

Monitors baby through video and audio, tracking sleep patterns and sending notifications.
"""
import yaml
import logging
import sys
import os
import signal
from datetime import datetime
from dotenv import load_dotenv

from database import DatabaseManager
from notifiers import NotificationManager
from monitors import VideoMonitor, AudioMonitor
from monitors.recording_manager import RecordingManager

# Load environment variables
load_dotenv()


def setup_logging(config):
    """
    Setup logging configuration.

    Args:
        config: Configuration dictionary
    """
    log_level = config['logging']['level']
    log_file = config['logging']['file']

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override with environment variables if present
        if os.getenv('PUSHBULLET_API_KEY'):
            config['notifications']['pushbullet']['api_key'] = os.getenv('PUSHBULLET_API_KEY')
        if os.getenv('WEBHOOK_URL'):
            config['notifications']['webhook']['url'] = os.getenv('WEBHOOK_URL')

        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


class BabyMonitor:
    """Main baby monitor application"""

    def __init__(self, config):
        """
        Initialize baby monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.db = DatabaseManager(config['database']['path'])
        self.notifier = NotificationManager(config['notifications'])
        self.video_monitor = VideoMonitor(config, self.db, self.notifier)
        self.audio_monitor = AudioMonitor(config, self.db, self.notifier)

        # Initialize recording manager if video storage is enabled
        self.recording_manager = None
        if config.get('video_storage', {}).get('enabled', False):
            storage_path = config['video_storage']['storage_path']
            retention_days = config['video_storage']['retention_days']
            self.recording_manager = RecordingManager(storage_path, retention_days)

        # Web interface
        self.web_app = None
        if config['web']['enabled']:
            from web.app import create_app
            self.web_app = create_app(self)

        self.running = False
        self.cleanup_thread = None

    def start(self):
        """Start all monitoring components"""
        self.logger.info("="*60)
        self.logger.info("Starting AI Baby Monitor")
        self.logger.info("="*60)

        # Start video monitoring
        if not self.video_monitor.start():
            self.logger.error("Failed to start video monitor")
            return False

        # Start audio monitoring
        if not self.audio_monitor.start():
            self.logger.warning("Audio monitor not started (may not be available)")

        self.running = True

        # Start recording cleanup task if recording manager is enabled
        if self.recording_manager:
            from threading import Thread
            import time

            def cleanup_task():
                """Periodic cleanup of old recordings"""
                while self.running:
                    # Run cleanup every 6 hours
                    time.sleep(6 * 60 * 60)
                    if self.running:
                        self.logger.info("Running periodic recording cleanup...")
                        deleted, freed = self.recording_manager.cleanup_old_recordings()
                        if deleted > 0:
                            self.logger.info(f"Cleanup: {deleted} files removed, {freed / 1024 / 1024:.2f} MB freed")

            self.cleanup_thread = Thread(target=cleanup_task, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("Recording cleanup task started")

        # Start web interface if enabled
        if self.web_app:
            from threading import Thread
            web_config = self.config['web']
            self.logger.info(f"Starting web interface on http://{web_config['host']}:{web_config['port']}")

            web_thread = Thread(
                target=self.web_app.run,
                kwargs={
                    'host': web_config['host'],
                    'port': web_config['port'],
                    'debug': web_config['debug'],
                    'use_reloader': False
                },
                daemon=True
            )
            web_thread.start()

        self.logger.info("AI Baby Monitor is running")
        self.logger.info("Press Ctrl+C to stop")

        return True

    def stop(self):
        """Stop all monitoring components"""
        if not self.running:
            return

        self.logger.info("Stopping AI Baby Monitor...")

        self.video_monitor.stop()
        self.audio_monitor.stop()
        self.db.close()

        self.running = False
        self.logger.info("AI Baby Monitor stopped")

    def get_status(self):
        """Get status of all monitoring components"""
        return {
            'running': self.running,
            'video': self.video_monitor.get_status(),
            'audio': self.audio_monitor.get_status(),
            'current_sleep_session': self.db.get_current_sleep_session()
        }


def main():
    """Main entry point"""
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)

    # Create baby monitor instance
    monitor = BabyMonitor(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start monitoring
    if not monitor.start():
        logger.error("Failed to start baby monitor")
        sys.exit(1)

    # Keep main thread alive
    try:
        signal.pause()
    except AttributeError:
        # signal.pause() not available on Windows
        import time
        while monitor.running:
            time.sleep(1)


if __name__ == '__main__':
    main()
