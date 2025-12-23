# app/tools/communication.py

from langchain_core.tools import tool
import json
from datetime import datetime

@tool
def ask_another_agent(agent_name: str, question: str) -> str:
    """
    Request help from another specialist agent.
    
    Use this when you need information or analysis from another agent:
    - Use "sql" for data queries and analytics
    - Use "negotiator" for supplier strategy and business context
    
    Args:
        agent_name: Which agent to ask ("sql" or "negotiator")
        question: Your question or request
    
    Returns:
        Routing instruction (processed by workflow)
    
    Examples:
        ask_another_agent("sql", "What's our YTD spend on bearings?")
        ask_another_agent("negotiator", "Is 60% spend concentration risky?")
    """
    return json.dumps({
        "action": "route_to",
        "agent": agent_name,
        "question": question,
        "timestamp": datetime.utcnow().isoformat()
    })