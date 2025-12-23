# app/workflow/state.py

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Simple state for multi-agent system.
    All communication flows through messages.
    """
    
    # Conversation history (all A2A communication preserved here)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Routing control
    next_agent: str  # "orchestrator" | "negotiator" | "sql" | "END"
    
    # Metadata
    session_id: str
    user_id: str
    timestamp: str