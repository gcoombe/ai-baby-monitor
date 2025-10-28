"""
Notification manager for sending alerts about baby monitor events.
"""
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications for baby monitor events"""

    def __init__(self, config):
        """
        Initialize notification manager.

        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.methods = config.get('methods', ['console'])
        self.event_config = config.get('events', {})

        # Initialize Pushbullet if configured
        self.pushbullet_api_key = config.get('pushbullet', {}).get('api_key')
        if 'pushbullet' in self.methods and not self.pushbullet_api_key:
            logger.warning("Pushbullet enabled but no API key provided")

        # Webhook URL
        self.webhook_url = config.get('webhook', {}).get('url')
        if 'webhook' in self.methods and not self.webhook_url:
            logger.warning("Webhook enabled but no URL provided")

    def should_notify(self, event_type):
        """
        Check if notifications should be sent for this event type.

        Args:
            event_type: Type of event

        Returns:
            bool: Whether to send notification
        """
        if not self.enabled:
            return False
        return self.event_config.get(event_type, True)

    def send_notification(self, title, message, event_type=None):
        """
        Send notification through configured methods.

        Args:
            title: Notification title
            message: Notification message
            event_type: Type of event (for filtering)

        Returns:
            bool: Whether any notification was sent successfully
        """
        if event_type and not self.should_notify(event_type):
            return False

        success = False

        for method in self.methods:
            try:
                if method == 'console':
                    self._send_console(title, message)
                    success = True
                elif method == 'pushbullet':
                    success = self._send_pushbullet(title, message) or success
                elif method == 'webhook':
                    success = self._send_webhook(title, message, event_type) or success
                else:
                    logger.warning(f"Unknown notification method: {method}")
            except Exception as e:
                logger.error(f"Error sending notification via {method}: {e}")

        return success

    def _send_console(self, title, message):
        """Print notification to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[{timestamp}] {title}")
        print(f"{message}")
        print(f"{'='*60}\n")

    def _send_pushbullet(self, title, message):
        """
        Send notification via Pushbullet.

        Args:
            title: Notification title
            message: Notification message

        Returns:
            bool: Whether notification was sent successfully
        """
        if not self.pushbullet_api_key:
            logger.warning("Cannot send Pushbullet notification: No API key")
            return False

        try:
            url = "https://api.pushbullet.com/v2/pushes"
            headers = {
                "Access-Token": self.pushbullet_api_key,
                "Content-Type": "application/json"
            }
            data = {
                "type": "note",
                "title": title,
                "body": message
            }

            response = requests.post(url, json=data, headers=headers, timeout=10)
            response.raise_for_status()

            logger.info("Pushbullet notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Pushbullet notification: {e}")
            return False

    def _send_webhook(self, title, message, event_type=None):
        """
        Send notification via custom webhook.

        Args:
            title: Notification title
            message: Notification message
            event_type: Type of event

        Returns:
            bool: Whether notification was sent successfully
        """
        if not self.webhook_url:
            logger.warning("Cannot send webhook notification: No URL configured")
            return False

        try:
            data = {
                "title": title,
                "message": message,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat()
            }

            response = requests.post(self.webhook_url, json=data, timeout=10)
            response.raise_for_status()

            logger.info("Webhook notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    def notify_baby_awake(self, confidence=None):
        """Send notification that baby is awake"""
        title = "Baby is Awake!"
        message = "Your baby has woken up and is active."
        if confidence:
            message += f"\nConfidence: {confidence:.0%}"
        return self.send_notification(title, message, 'baby_awake')

    def notify_baby_stirring(self, confidence=None):
        """Send notification that baby is stirring"""
        title = "Baby Stirring"
        message = "Your baby is stirring but may not be fully awake."
        if confidence:
            message += f"\nConfidence: {confidence:.0%}"
        return self.send_notification(title, message, 'baby_stirring')

    def notify_cry_detected(self, confidence=None):
        """Send notification that crying was detected"""
        title = "Crying Detected!"
        message = "Your baby may be crying."
        if confidence:
            message += f"\nConfidence: {confidence:.0%}"
        return self.send_notification(title, message, 'cry_detected')

    def notify_long_wake_period(self, duration_minutes):
        """Send notification about extended wake period"""
        title = "Extended Wake Period"
        message = f"Your baby has been awake for {duration_minutes:.0f} minutes."
        return self.send_notification(title, message, 'long_wake_period')
