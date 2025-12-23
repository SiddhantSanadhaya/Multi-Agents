# app/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Request schema for procurement queries"""
    
    query: str = Field(..., description="User's procurement question")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Find the best bearing suppliers in the US",
                "user_id": "user123"
            }
        }


class AgentStep(BaseModel):
    """Individual agent step in the conversation"""
    
    step_number: int
    agent_name: str
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


class QueryResponse(BaseModel):
    """Response schema with full trace"""
    
    session_id: str
    answer: str
    agent_trace: List[AgentStep]
    total_agents_involved: int
    execution_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_123",
                "answer": "Based on analysis, SupplierX is recommended...",
                "agent_trace": [
                    {
                        "step_number": 1,
                        "agent_name": "orchestrator",
                        "thought": "I need supplier analysis",
                        "action": "ask_another_agent",
                        "action_input": {"agent_name": "negotiator", "question": "..."}
                    }
                ],
                "total_agents_involved": 2,
                "execution_time_seconds": 3.45,
                "timestamp": "2024-12-17T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    version: str
    timestamp: datetime


class SessionResponse(BaseModel):
    """Session trace response"""
    
    session_id: str
    total_interactions: int
    logs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class SessionListResponse(BaseModel):
    """List of recent sessions"""
    
    sessions: List[Dict[str, Any]]
    total_count: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime