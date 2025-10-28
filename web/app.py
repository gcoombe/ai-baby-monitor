"""
Web dashboard for baby monitor.
"""
from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
import cv2
import logging

logger = logging.getLogger(__name__)


def create_app(baby_monitor):
    """
    Create Flask application.

    Args:
        baby_monitor: BabyMonitor instance

    Returns:
        Flask app
    """
    app = Flask(__name__)
    CORS(app)

    # Store reference to baby monitor
    app.baby_monitor = baby_monitor

    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html')

    @app.route('/api/status')
    def get_status():
        """Get current monitoring status"""
        try:
            status = baby_monitor.get_status()

            # Add sleep session info
            current_session = baby_monitor.db.get_current_sleep_session()
            if current_session:
                from datetime import datetime
                duration = (datetime.now() - current_session.start_time).total_seconds() / 60
                status['current_sleep_session'] = {
                    'start_time': current_session.start_time.isoformat(),
                    'duration_minutes': duration,
                    'stir_count': current_session.stir_count
                }
            else:
                status['current_sleep_session'] = None

            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/events')
    def get_events():
        """Get recent events"""
        try:
            limit = 50
            events = baby_monitor.db.get_recent_events(limit)

            events_list = [{
                'id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'confidence': event.confidence,
                'details': event.details,
                'notification_sent': event.notification_sent
            } for event in events]

            return jsonify(events_list)
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/sleep-sessions')
    def get_sleep_sessions():
        """Get recent sleep sessions"""
        try:
            limit = 10
            sessions = baby_monitor.db.get_recent_sleep_sessions(limit)

            sessions_list = [{
                'id': session.id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'duration_minutes': session.duration_minutes,
                'quality_score': session.quality_score,
                'stir_count': session.stir_count
            } for session in sessions]

            return jsonify(sessions_list)
        except Exception as e:
            logger.error(f"Error getting sleep sessions: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/statistics')
    def get_statistics():
        """Get sleep statistics"""
        try:
            days = 7
            stats = baby_monitor.db.get_sleep_statistics(days)
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/video-feed')
    def video_feed():
        """Video streaming route"""
        def generate():
            while True:
                frame = baby_monitor.video_monitor.get_current_frame()
                if frame is None:
                    continue

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app
