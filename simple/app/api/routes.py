# app/api/routes.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.api.schemas import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse, 
    SessionResponse,
    SessionListResponse,
    ErrorResponse
)
from app.workflow.graph import workflow_app
from app.services.logger import (
    get_session_logs, 
    get_session_metadata,
    get_recent_sessions,
    create_session_metadata,
    update_session_metadata
)
from app.services.database import create_tables
from datetime import datetime
import uuid
import time

router = APIRouter()

# Initialize database tables on startup
create_tables()


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process a procurement query through the multi-agent system.
    
    The system will:
    1. Route through orchestrator
    2. Dynamically involve specialist agents
    3. Return answer with full trace
    """
    
    # Generate session ID if not provided
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    
    try:
        # Track execution time
        start_time = time.time()
        
        # Create session metadata
        background_tasks.add_task(
            create_session_metadata,
            session_id=session_id,
            user_id=request.user_id or "anonymous",
            start_time=datetime.utcnow()
        )
        
        # Prepare state
        initial_state = {
            "messages": [("user", request.query)],
            "next_agent": "orchestrator",
            "session_id": session_id,
            "user_id": request.user_id or "anonymous",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Invoke workflow
        print(f"\n{'='*80}")
        print(f"Processing query: {request.query}")
        print(f"Session: {session_id}")
        print(f"{'='*80}\n")
        
        result = workflow_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract final answer
        final_answer = result["messages"][-1].content
        
        # Retrieve full trace from logs
        session_logs = get_session_logs(session_id)
        
        # Build agent trace
        agent_trace = []
        agents_involved = set()
        
        for log in session_logs:
            agents_involved.add(log["agent_name"])
            for step in log["react_trace"]:
                agent_trace.append({
                    "step_number": step["step_number"],
                    "agent_name": log["agent_name"],
                    "thought": step["thought"],
                    "action": step.get("action"),
                    "action_input": step.get("action_input"),
                    "observation": step.get("observation")
                })
        
        # Update session metadata in background
        background_tasks.add_task(
            update_session_metadata,
            session_id=session_id,
            final_answer=final_answer,
            end_time=datetime.utcnow(),
            total_turns=len(session_logs)
        )
        
        return QueryResponse(
            session_id=session_id,
            answer=final_answer,
            agent_trace=agent_trace,
            total_agents_involved=len(agents_involved),
            execution_time_seconds=round(execution_time, 2),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_trace(session_id: str):
    """
    Retrieve full trace for a session.
    Useful for debugging and auditing.
    """
    
    try:
        logs = get_session_logs(session_id)
        metadata = get_session_metadata(session_id)
        
        if not logs and not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return SessionResponse(
            session_id=session_id,
            total_interactions=len(logs),
            logs=logs,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session: {str(e)}"
        )


@router.get("/sessions", response_model=SessionListResponse)
async def get_recent_sessions_list(limit: int = 10):
    """
    Get list of recent sessions for analytics.
    """
    
    try:
        sessions = get_recent_sessions(limit=limit)
        
        return SessionListResponse(
            sessions=sessions,
            total_count=len(sessions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sessions: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )


@router.get("/debug/session/{session_id}")
async def debug_session(session_id: str):
    """
    Debug endpoint with detailed session information.
    Returns raw data for troubleshooting.
    """
    
    try:
        logs = get_session_logs(session_id)
        metadata = get_session_metadata(session_id)
        
        if not logs and not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return {
            "session_id": session_id,
            "metadata": metadata,
            "logs": logs,
            "debug_info": {
                "total_interactions": len(logs),
                "agents_used": list(set(log["agent_name"] for log in logs)),
                "total_steps": sum(len(log["react_trace"]) for log in logs),
                "session_duration_estimate": None  # Could calculate from timestamps
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Debug error: {str(e)}"
        )