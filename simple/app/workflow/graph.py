# app/workflow/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.workflow.state import AgentState
from app.agents.orchestrator import create_orchestrator_agent
from app.agents.negotiator import create_negotiator_agent
from app.agents.sql_agent import create_sql_agent
from app.services.logger import log_interaction
import json

# Initialize agents (singletons)
orchestrator = create_orchestrator_agent()
negotiator = create_negotiator_agent()
sql_agent = create_sql_agent()


def orchestrator_node(state: AgentState) -> AgentState:
    """Orchestrator processes and routes"""
    
    print(f"\n{'='*60}")
    print("🎯 ORCHESTRATOR")
    print(f"{'='*60}\n")
    
    result = orchestrator.invoke({"messages": state["messages"]})
    
    # Extract routing decision
    next_agent = extract_routing(result["messages"][-1])
    
    # Log interaction
    log_interaction(
        session_id=state["session_id"],
        agent_name="orchestrator",
        messages=result["messages"],
        next_agent=next_agent
    )
    
    state["messages"] = result["messages"]
    state["next_agent"] = next_agent
    
    return state


def negotiator_node(state: AgentState) -> AgentState:
    """Negotiator analyzes, may request SQL help"""
    
    print(f"\n{'='*60}")
    print("🤝 NEGOTIATOR")
    print(f"{'='*60}\n")
    
    result = negotiator.invoke({"messages": state["messages"]})
    
    # Check if negotiator wants to route to SQL
    next_agent = extract_routing(result["messages"][-1])
    
    # Log interaction
    log_interaction(
        session_id=state["session_id"],
        agent_name="negotiator",
        messages=result["messages"],
        next_agent=next_agent
    )
    
    state["messages"] = result["messages"]
    # If no routing requested, return to orchestrator
    state["next_agent"] = next_agent if next_agent else "orchestrator"
    
    return state


def sql_node(state: AgentState) -> AgentState:
    """SQL analyzes data, may request Negotiator context"""
    
    print(f"\n{'='*60}")
    print("📊 SQL")
    print(f"{'='*60}\n")
    
    result = sql_agent.invoke({"messages": state["messages"]})
    
    # Check if SQL wants to route to Negotiator
    next_agent = extract_routing(result["messages"][-1])
    
    # Log interaction
    log_interaction(
        session_id=state["session_id"],
        agent_name="sql",
        messages=result["messages"],
        next_agent=next_agent
    )
    
    state["messages"] = result["messages"]
    # If no routing requested, return to orchestrator
    state["next_agent"] = next_agent if next_agent else "orchestrator"
    
    return state


def extract_routing(message) -> str:
    """
    Extract routing decision from agent's tool calls.
    
    Looks for ask_another_agent tool call.
    Returns agent name or "END" if no routing requested.
    """
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call["name"] == "ask_another_agent":
                agent_name = tool_call["args"]["agent_name"]
                print(f"  → Routing to: {agent_name}")
                return agent_name
    
    print(f"  → No routing requested, task complete")
    return "END"


def route_next(state: AgentState) -> str:
    """Simple routing based on next_agent field"""
    next_agent = state.get("next_agent", "END")
    return next_agent


def create_workflow():
    """
    Create the simple multi-agent workflow.
    
    Key features:
    - Non-deterministic routing
    - Each agent can call any other agent
    - Full A2A communication through messages
    """
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("negotiator", negotiator_node)
    workflow.add_node("sql", sql_node)
    
    # Start with orchestrator
    workflow.set_entry_point("orchestrator")
    
    # ✅ KEY: Conditional edges from EACH agent to ALL agents
    # This enables non-deterministic, dynamic routing
    for agent in ["orchestrator", "negotiator", "sql"]:
        workflow.add_conditional_edges(
            agent,
            route_next,
            {
                "orchestrator": "orchestrator",
                "negotiator": "negotiator",
                "sql": "sql",
                "END": END
            }
        )
    
    # Compile with memory for debugging
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Global workflow instance
workflow_app = create_workflow()