"""
Database manager for tracking baby monitor events and sleep patterns.
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()


class SleepSession(Base):
    """Tracks sleep sessions"""
    __tablename__ = 'sleep_sessions'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # Based on stirring frequency
    stir_count = Column(Integer, default=0)


class Event(Base):
    """Tracks all baby monitor events"""
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    event_type = Column(String(50), nullable=False)  # motion, cry, awake, asleep, stirring
    confidence = Column(Float, nullable=True)
    details = Column(String(500), nullable=True)
    notification_sent = Column(Boolean, default=False)


class DatabaseManager:
    """Manages database operations for the baby monitor"""

    def __init__(self, db_path):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.current_sleep_session = None

    def start_sleep_session(self):
        """Start a new sleep session"""
        # End any existing session first
        if self.current_sleep_session:
            self.end_sleep_session()

        self.current_sleep_session = SleepSession(start_time=datetime.now())
        self.session.add(self.current_sleep_session)
        self.session.commit()
        return self.current_sleep_session.id

    def end_sleep_session(self):
        """End the current sleep session"""
        if not self.current_sleep_session:
            return None

        self.current_sleep_session.end_time = datetime.now()
        duration = (self.current_sleep_session.end_time -
                   self.current_sleep_session.start_time).total_seconds() / 60
        self.current_sleep_session.duration_minutes = duration

        # Calculate quality score (100 - penalty for stirring)
        stir_penalty = min(self.current_sleep_session.stir_count * 5, 50)
        self.current_sleep_session.quality_score = max(100 - stir_penalty, 0)

        self.session.commit()
        session_id = self.current_sleep_session.id
        self.current_sleep_session = None
        return session_id

    def record_stir(self):
        """Record a stirring event in the current sleep session"""
        if self.current_sleep_session:
            self.current_sleep_session.stir_count += 1
            self.session.commit()

    def log_event(self, event_type, confidence=None, details=None, notification_sent=False):
        """
        Log a baby monitor event.

        Args:
            event_type: Type of event (motion, cry, awake, asleep, stirring)
            confidence: Confidence score (0-1)
            details: Additional details about the event
            notification_sent: Whether a notification was sent for this event
        """
        event = Event(
            timestamp=datetime.now(),
            event_type=event_type,
            confidence=confidence,
            details=details,
            notification_sent=notification_sent
        )
        self.session.add(event)
        self.session.commit()
        return event.id

    def get_recent_events(self, limit=50):
        """Get recent events"""
        return self.session.query(Event).order_by(
            Event.timestamp.desc()
        ).limit(limit).all()

    def get_recent_sleep_sessions(self, limit=10):
        """Get recent sleep sessions"""
        return self.session.query(SleepSession).order_by(
            SleepSession.start_time.desc()
        ).limit(limit).all()

    def get_current_sleep_session(self):
        """Get the current active sleep session"""
        return self.current_sleep_session

    def get_sleep_statistics(self, days=7):
        """Get sleep statistics for the past N days"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)

        sessions = self.session.query(SleepSession).filter(
            SleepSession.start_time >= cutoff,
            SleepSession.end_time.isnot(None)
        ).all()

        if not sessions:
            return {
                'total_sessions': 0,
                'average_duration': 0,
                'average_quality': 0,
                'total_sleep_time': 0
            }

        return {
            'total_sessions': len(sessions),
            'average_duration': sum(s.duration_minutes for s in sessions) / len(sessions),
            'average_quality': sum(s.quality_score for s in sessions) / len(sessions),
            'total_sleep_time': sum(s.duration_minutes for s in sessions)
        }

    def close(self):
        """Close database connection"""
        self.session.close()
