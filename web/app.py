"""
Web dashboard for baby monitor.
"""
from flask import Flask, render_template, jsonify, Response, send_file, request
from flask_cors import CORS
import cv2
import logging
import os

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

    @app.route('/api/recordings')
    def get_recordings():
        """Get list of available recordings"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            # Get query parameters
            limit = request.args.get('limit', default=None, type=int)
            sort_by = request.args.get('sort_by', default='date', type=str)
            ascending = request.args.get('ascending', default='false', type=str).lower() == 'true'

            recordings = baby_monitor.recording_manager.list_recordings(
                limit=limit,
                sort_by=sort_by,
                ascending=ascending
            )

            # Convert datetime objects to ISO format
            for recording in recordings:
                recording['created'] = recording['created'].isoformat()
                recording['modified'] = recording['modified'].isoformat()

            return jsonify(recordings)
        except Exception as e:
            logger.error(f"Error getting recordings: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recordings/stats')
    def get_recording_stats():
        """Get storage statistics for recordings"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            stats = baby_monitor.recording_manager.get_storage_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting recording stats: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recordings/<filename>', methods=['DELETE'])
    def delete_recording(filename):
        """Delete a specific recording"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            success = baby_monitor.recording_manager.delete_recording(filename)
            if success:
                return jsonify({'message': f'Recording {filename} deleted successfully'})
            else:
                return jsonify({'error': 'Failed to delete recording'}), 400
        except Exception as e:
            logger.error(f"Error deleting recording: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recordings/<filename>/download')
    def download_recording(filename):
        """Download a recording file"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            file_path = baby_monitor.recording_manager.get_recording_path(filename)
            if not file_path:
                return jsonify({'error': 'Recording not found'}), 404

            return send_file(
                file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='video/mp4'
            )
        except Exception as e:
            logger.error(f"Error downloading recording: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recordings/<filename>/stream')
    def stream_recording(filename):
        """Stream a recording file"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            file_path = baby_monitor.recording_manager.get_recording_path(filename)
            if not file_path:
                return jsonify({'error': 'Recording not found'}), 404

            return send_file(
                file_path,
                mimetype='video/mp4'
            )
        except Exception as e:
            logger.error(f"Error streaming recording: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recordings/cleanup', methods=['POST'])
    def manual_cleanup():
        """Manually trigger cleanup of old recordings"""
        try:
            if not baby_monitor.recording_manager:
                return jsonify({'error': 'Video storage not enabled'}), 404

            deleted, freed = baby_monitor.recording_manager.cleanup_old_recordings()
            return jsonify({
                'files_deleted': deleted,
                'bytes_freed': freed,
                'mb_freed': freed / 1024 / 1024
            })
        except Exception as e:
            logger.error(f"Error during manual cleanup: {e}")
            return jsonify({'error': str(e)}), 500

    return app
