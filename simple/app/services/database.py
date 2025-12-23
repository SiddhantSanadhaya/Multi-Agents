# app/services/database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from app.config import get_settings
from typing import Generator

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class AgentInteraction(Base):
    """Model for storing agent interaction logs"""
    
    __tablename__ = "agent_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    react_trace = Column(JSON, nullable=False)
    next_agent = Column(String(50))
    message_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionMetadata(Base):
    """Model for storing session metadata"""
    
    __tablename__ = "session_metadata"
    
    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    total_turns = Column(Integer)
    final_answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session for direct use"""
    return SessionLocal()