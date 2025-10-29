"""
Recording management utilities for handling video file retention and cleanup.
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class RecordingManager:
    """Manages video recordings including retention policies and file operations"""

    def __init__(self, storage_path, retention_days=7):
        """
        Initialize recording manager.

        Args:
            storage_path: Directory where recordings are stored
            retention_days: Number of days to keep recordings (0 = keep forever)
        """
        self.storage_path = storage_path
        self.retention_days = retention_days

        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)

    def cleanup_old_recordings(self):
        """
        Delete recordings older than retention period.

        Returns:
            tuple: (number of files deleted, bytes freed)
        """
        if self.retention_days == 0:
            logger.debug("Retention disabled (retention_days=0), skipping cleanup")
            return 0, 0

        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        files_deleted = 0
        bytes_freed = 0

        try:
            for file_path in Path(self.storage_path).glob("recording_*.mp4"):
                # Get file modification time
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_mtime < cutoff_time:
                    file_size = file_path.stat().st_size
                    try:
                        file_path.unlink()
                        files_deleted += 1
                        bytes_freed += file_size
                        logger.info(f"Deleted old recording: {file_path.name}")
                    except OSError as e:
                        logger.error(f"Failed to delete {file_path.name}: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        if files_deleted > 0:
            logger.info(f"Cleanup complete: {files_deleted} files deleted, {bytes_freed / 1024 / 1024:.2f} MB freed")

        return files_deleted, bytes_freed

    def list_recordings(self, limit=None, sort_by='date', ascending=False):
        """
        List available recordings.

        Args:
            limit: Maximum number of recordings to return (None = all)
            sort_by: Sort criterion ('date', 'size', 'name')
            ascending: Sort order (False = descending)

        Returns:
            list: List of recording info dictionaries
        """
        recordings = []

        try:
            for file_path in Path(self.storage_path).glob("recording_*.mp4"):
                stat = file_path.stat()
                recordings.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
                })

        except Exception as e:
            logger.error(f"Error listing recordings: {e}")

        # Sort recordings
        if sort_by == 'date':
            recordings.sort(key=lambda x: x['modified'], reverse=not ascending)
        elif sort_by == 'size':
            recordings.sort(key=lambda x: x['size'], reverse=not ascending)
        elif sort_by == 'name':
            recordings.sort(key=lambda x: x['filename'], reverse=not ascending)

        # Apply limit
        if limit:
            recordings = recordings[:limit]

        return recordings

    def get_storage_stats(self):
        """
        Get storage statistics for recordings.

        Returns:
            dict: Storage statistics
        """
        total_size = 0
        file_count = 0
        oldest_file = None
        newest_file = None

        try:
            for file_path in Path(self.storage_path).glob("recording_*.mp4"):
                stat = file_path.stat()
                total_size += stat.st_size
                file_count += 1

                file_mtime = datetime.fromtimestamp(stat.st_mtime)
                if oldest_file is None or file_mtime < oldest_file:
                    oldest_file = file_mtime
                if newest_file is None or file_mtime > newest_file:
                    newest_file = file_mtime

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")

        return {
            'total_files': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'total_size_gb': total_size / 1024 / 1024 / 1024,
            'oldest_recording': oldest_file.isoformat() if oldest_file else None,
            'newest_recording': newest_file.isoformat() if newest_file else None,
            'storage_path': self.storage_path
        }

    def delete_recording(self, filename):
        """
        Delete a specific recording.

        Args:
            filename: Name of the recording file to delete

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        file_path = Path(self.storage_path) / filename

        # Security check: ensure filename is just a filename, not a path
        if filename != file_path.name or '..' in filename:
            logger.error(f"Invalid filename: {filename}")
            return False

        # Ensure it's a recording file
        if not filename.startswith('recording_') or not filename.endswith('.mp4'):
            logger.error(f"Not a recording file: {filename}")
            return False

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted recording: {filename}")
                return True
            else:
                logger.warning(f"Recording not found: {filename}")
                return False
        except OSError as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False

    def get_recording_path(self, filename):
        """
        Get the full path to a recording file.

        Args:
            filename: Name of the recording file

        Returns:
            str: Full path to recording or None if invalid
        """
        file_path = Path(self.storage_path) / filename

        # Security check
        if filename != file_path.name or '..' in filename:
            logger.error(f"Invalid filename: {filename}")
            return None

        if not file_path.exists():
            return None

        return str(file_path)
