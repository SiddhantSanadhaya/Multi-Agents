# app/services/logger.py

from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.services.database import get_db_session, AgentInteraction, SessionMetadata


def log_interaction(
    session_id: str,
    agent_name: str,
    messages: List,
    next_agent: str
):
    """
    Log agent interaction for debugging.
    
    Stores:
    - Full ReAct trace (thoughts, actions, observations)
    - Routing decisions
    - Timestamps
    - Token usage (if available)
    """
    
    db = get_db_session()
    
    try:
        # Extract ReAct trace
        react_trace = extract_react_trace(messages)
        
        # Create log entry
        interaction = AgentInteraction(
            session_id=session_id,
            agent_name=agent_name,
            timestamp=datetime.utcnow(),
            react_trace=react_trace,
            next_agent=next_agent,
            message_count=len(messages)
        )
        
        db.add(interaction)
        db.commit()
        
        # Also log to console for development
        print(f"\n📝 Logged: {agent_name} → {next_agent}")
        print(f"   Steps: {len(react_trace)}")
        
    except Exception as e:
        print(f"Error logging interaction: {e}")
        db.rollback()
    finally:
        db.close()


def extract_react_trace(messages: List) -> List[Dict[str, Any]]:
    """
    Extract ReAct steps from messages.
    
    Returns list of: {thought, action, action_input, observation}
    """
    trace = []
    
    for i, message in enumerate(messages):
        # Check for tool calls (Actions)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                step = {
                    "step_number": len(trace) + 1,
                    "thought": message.content if message.content else "Planning action...",
                    "action": tool_call["name"],
                    "action_input": tool_call["args"],
                    "observation": None  # Will be filled by next message
                }
                trace.append(step)
        
        # Check for tool responses (Observations)
        elif hasattr(message, 'name') and message.name:
            # This is a tool response
            if trace and trace[-1]["observation"] is None:
                trace[-1]["observation"] = message.content
    
    return trace


def get_session_logs(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve all logs for a session"""
    
    db = get_db_session()
    
    try:
        interactions = db.query(AgentInteraction).filter(
            AgentInteraction.session_id == session_id
        ).order_by(AgentInteraction.timestamp).all()
        
        logs = []
        for interaction in interactions:
            logs.append({
                "agent_name": interaction.agent_name,
                "timestamp": interaction.timestamp.isoformat(),
                "react_trace": interaction.react_trace,
                "next_agent": interaction.next_agent,
                "message_count": interaction.message_count
            })
        
        return logs
        
    finally:
        db.close()


def create_session_metadata(
    session_id: str,
    user_id: str = None,
    start_time: datetime = None
) -> SessionMetadata:
    """Create session metadata entry"""
    
    db = get_db_session()
    
    try:
        metadata = SessionMetadata(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time or datetime.utcnow(),
            total_turns=0
        )
        
        db.add(metadata)
        db.commit()
        db.refresh(metadata)
        
        return metadata
        
    except Exception as e:
        print(f"Error creating session metadata: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def update_session_metadata(
    session_id: str,
    final_answer: str = None,
    end_time: datetime = None,
    total_turns: int = None
):
    """Update session metadata when session completes"""
    
    db = get_db_session()
    
    try:
        metadata = db.query(SessionMetadata).filter(
            SessionMetadata.session_id == session_id
        ).first()
        
        if metadata:
            if final_answer:
                metadata.final_answer = final_answer
            if end_time:
                metadata.end_time = end_time
            if total_turns is not None:
                metadata.total_turns = total_turns
                
            db.commit()
            
    except Exception as e:
        print(f"Error updating session metadata: {e}")
        db.rollback()
    finally:
        db.close()


def get_session_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session metadata"""
    
    db = get_db_session()
    
    try:
        metadata = db.query(SessionMetadata).filter(
            SessionMetadata.session_id == session_id
        ).first()
        
        if metadata:
            return {
                "session_id": metadata.session_id,
                "user_id": metadata.user_id,
                "start_time": metadata.start_time.isoformat() if metadata.start_time else None,
                "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
                "total_turns": metadata.total_turns,
                "final_answer": metadata.final_answer,
                "created_at": metadata.created_at.isoformat()
            }
        
        return None
        
    finally:
        db.close()


def get_recent_sessions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent sessions for analytics"""
    
    db = get_db_session()
    
    try:
        sessions = db.query(SessionMetadata).order_by(
            desc(SessionMetadata.created_at)
        ).limit(limit).all()
        
        return [
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "total_turns": session.total_turns,
                "created_at": session.created_at.isoformat()
            }
            for session in sessions
        ]
        
    finally:
        db.close()