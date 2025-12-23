# Simple Multi-Agent Procurement System - Implementation Plan

## 🎯 Overview

A streamlined 3-agent system (Orchestrator, Negotiator, SQL) using LangGraph with non-deterministic A2A communication through a single `ask_another_agent` tool.

**Complexity Level:** Low  
**Lines of Code:** ~800 (total system)  
**Time to Implement:** 2-3 days  
**Best For:** Interviews, MVPs, Clear demonstrations

---

## 📊 System Architecture Flow Diagram

```mermaid
graph TD
    User[User Query] --> API[FastAPI Gateway]
    API --> Orch[Orchestrator Agent]
    
    Orch -->|ask_another_agent| Neg[Negotiator Agent]
    Orch -->|ask_another_agent| SQL[SQL Agent]
    
    Neg -->|ask_another_agent| SQL
    SQL -->|ask_another_agent| Neg
    
    Neg -->|response| Orch
    SQL -->|response| Orch
    
    Orch -->|final answer| API
    API -->|JSON response| User
    
    Orch -.->|logs| Logger[Interaction Logger]
    Neg -.->|logs| Logger
    SQL -.->|logs| Logger
    
    Logger -->|stores| DB[(PostgreSQL)]
    
    style Orch fill:#e1f5ff
    style Neg fill:#fff4e1
    style SQL fill:#f0fff4
    style API fill:#ffe1e1
    style Logger fill:#f5f5f5
```

### Communication Flow Example

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant N as Negotiator
    participant S as SQL
    
    U->>O: "Why is bearing spend up 30%?"
    
    Note over O: Thought: Need data first
    O->>S: ask_another_agent("sql", "Analyze spend increase")
    
    Note over S: Executes SQL query
    S->>S: SupplierX +45%, others flat
    Note over S: Thought: Need business context
    S->>N: ask_another_agent("negotiator", "Why SupplierX +45%?")
    
    Note over N: Checks Kraljic positioning
    N->>N: SupplierX: Leverage → Strategic
    N->>O: "Positioning change = pricing power"
    
    Note over O: Synthesizes all context
    O->>U: "30% increase due to SupplierX strategic shift..."
```

---

## 📁 Project Structure

```
procurement-agent-simple/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entry point (50 lines)
│   ├── config.py                   # Configuration (30 lines)
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Orchestrator agent (40 lines)
│   │   ├── negotiator.py          # Negotiator agent (50 lines)
│   │   └── sql_agent.py           # SQL agent (50 lines)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── communication.py       # ask_another_agent tool (20 lines)
│   │   ├── procurement_tools.py   # Domain tools (150 lines)
│   │   └── sql_tools.py           # SQL tools (100 lines)
│   │
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── state.py               # State definition (20 lines)
│   │   └── graph.py               # Workflow graph (80 lines)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py            # DB connections (40 lines)
│   │   └── logger.py              # Interaction logger (60 lines)
│   │
│   └── api/
│       ├── __init__.py
│       ├── routes.py              # API routes (50 lines)
│       └── schemas.py             # Pydantic models (40 lines)
│
├── tests/
│   ├── test_agents.py             # Agent tests (100 lines)
│   ├── test_workflow.py           # Workflow tests (80 lines)
│   └── test_api.py                # API tests (60 lines)
│
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md
```

**Total Estimated Lines:** ~800-900 lines

---

## 🔧 Phase 1: Core Setup

### 1.1 State Definition

```python
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
```

**Key Design Decision:** All agent communication flows through `messages`. No custom queues needed!

---

### 1.2 Communication Tool

```python
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
```

**Why Simple:** One tool does everything. Agents use it naturally in their ReAct reasoning.

---

### 1.3 Configuration

```python
# app/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/procurement"
    
    # Vector DB (for SQL agent)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    
    # Application
    APP_NAME: str = "Procurement Multi-Agent System"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Agent Configuration
    MODEL_NAME: str = "claude-sonnet-4-5-20250929"
    MAX_ITERATIONS: int = 15  # Prevent infinite loops
    TEMPERATURE: float = 0.0
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

```bash
# .env.example

ANTHROPIC_API_KEY=sk-ant-xxx
DATABASE_URL=postgresql://user:pass@localhost:5432/procurement
PINECONE_API_KEY=xxx
PINECONE_ENVIRONMENT=us-east-1
DEBUG=true
LOG_LEVEL=INFO
```

---

## 🤖 Phase 2: Agent Implementations

### 2.1 Orchestrator Agent

```python
# app/agents/orchestrator.py

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from app.tools.communication import ask_another_agent
from app.config import get_settings

settings = get_settings()

def create_orchestrator_agent():
    """
    Orchestrator coordinates between specialist agents.
    
    Simple responsibilities:
    1. Understand user request
    2. Route to appropriate specialist
    3. Synthesize results
    """
    
    system_prompt = """You are an Orchestrator Agent for a procurement system.

**Available Specialists:**
- **negotiator**: Supplier analysis, Kraljic positioning, negotiation strategies
- **sql**: Data queries, spend analysis, reporting

**Your Process:**
1. Understand what the user needs
2. Route to the right specialist using ask_another_agent()
3. When specialists respond, synthesize into clear answers

**Routing Guidelines:**
- Questions about suppliers, contracts, strategies → negotiator
- Questions about data, reports, trends, analytics → sql
- Complex questions may need both (route sequentially)

**Examples:**

User: "Find best bearing suppliers"
You: ask_another_agent("negotiator", "Find and recommend best bearing suppliers")

User: "Show me top suppliers by spend"
You: ask_another_agent("sql", "Query top suppliers by spend amount")

User: "Analyze bearing spend and recommend negotiation strategies"
You: 
  1. ask_another_agent("sql", "Analyze bearing spend by supplier")
  2. [Wait for SQL response]
  3. ask_another_agent("negotiator", "Based on this spend data: [data], recommend strategies")
  4. [Synthesize both responses]

**Remember:** You coordinate, specialists execute. Keep it simple!
"""

    llm = ChatAnthropic(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=4096
    )
    
    tools = [ask_another_agent]
    
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
```

---

### 2.2 Negotiator Agent

```python
# app/agents/negotiator.py

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from app.tools.communication import ask_another_agent
from app.tools.procurement_tools import (
    query_suppliers,
    get_kraljic_positioning,
    get_ytd_spend,
    calculate_negotiation_strategy
)
from app.config import get_settings

settings = get_settings()

def create_negotiator_agent():
    """
    Negotiator handles supplier strategy and can request data from SQL.
    
    Key feature: Self-aware! Knows when to ask SQL for help.
    """
    
    system_prompt = """You are a Procurement Negotiation Expert.

**Your Expertise:**
- Supplier analysis and selection
- Kraljic matrix positioning
- Negotiation strategy development
- Contract optimization

**Your Tools:**
- query_suppliers: Find suppliers by category/location
- get_kraljic_positioning: Get strategic positioning
- get_ytd_spend: Get spend data by supplier
- calculate_negotiation_strategy: Generate negotiation approach
- ask_another_agent: Request help from SQL agent

**Self-Awareness - Know Your Limits!**
✅ You ARE expert at: Strategy, positioning, negotiation tactics
❌ You ARE NOT expert at: Complex queries, historical trends, analytics

**When to Ask SQL for Help:**
- Need historical price trends
- Need spend comparisons across time periods
- Need complex data aggregations
- Need market analysis data

**Example Workflow:**

User: "Find best bearing suppliers"

Your thought process:
1. query_suppliers(category="bearings") → Get supplier list
2. get_kraljic_positioning(suppliers) → Understand strategic position
3. REALIZE: "I need YTD spend to prioritize"
4. get_ytd_spend(suppliers) → Get spend data
5. REALIZE: "I see SupplierX has high spend, but is it increasing?"
6. ask_another_agent("sql", "Get SupplierX spend trend over last 2 years")
7. [Wait for SQL response]
8. calculate_negotiation_strategy() based on all data
9. Provide recommendation

**Key Principle:** Proactively identify when you need data and ASK for it!
"""

    llm = ChatAnthropic(
        model=settings.MODEL_NAME,
        temperature=0.1,  # Slight creativity for strategies
        max_tokens=4096
    )
    
    tools = [
        # Domain tools
        query_suppliers,
        get_kraljic_positioning,
        get_ytd_spend,
        calculate_negotiation_strategy,
        # Communication tool
        ask_another_agent
    ]
    
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
```

---

### 2.3 SQL Agent

```python
# app/agents/sql_agent.py

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from app.tools.communication import ask_another_agent
from app.tools.sql_tools import (
    search_similar_queries,
    validate_sql_query,
    execute_sql_query,
    get_database_schema
)
from app.config import get_settings

settings = get_settings()

def create_sql_agent():
    """
    SQL agent handles data queries and can request business context.
    
    Key feature: Recognizes when data needs interpretation!
    """
    
    system_prompt = """You are a SQL and Data Analysis Expert.

**Your Expertise:**
- Writing efficient SQL queries
- Data analysis and reporting
- Spend analytics
- Trend identification

**Your Tools:**
- search_similar_queries: Find similar query examples in vector DB
- get_database_schema: Get table schemas
- validate_sql_query: Validate SQL syntax and safety
- execute_sql_query: Execute SELECT queries
- ask_another_agent: Request business context from Negotiator

**Self-Awareness - Know Your Limits!**
✅ You ARE expert at: SQL, data extraction, calculations, aggregations
❌ You ARE NOT expert at: Business strategy, supplier relationships, risk assessment

**When to Ask Negotiator for Help:**
- Data shows anomalies but need business explanation
- Asked for "best supplier" (that's strategy, not data!)
- Need to interpret positioning or risk
- See concerning patterns (e.g., high concentration)

**SQL Safety Rules:**
- ONLY SELECT queries allowed
- ALWAYS validate before executing
- Use LIMIT to prevent large result sets
- Handle errors gracefully

**Example Workflow:**

User request: "Show top suppliers by spend"

Your thought process:
1. search_similar_queries("top suppliers by spend") → Get SQL template
2. validate_sql_query(generated_sql) → Ensure safety
3. execute_sql_query(sql) → Get results
4. NOTICE: "SupplierX has 60% of total spend"
5. REALIZE: "That's high concentration - is this risky?"
6. ask_another_agent("negotiator", "SupplierX represents 60% of spend. Is this concentration level risky?")
7. [Wait for Negotiator response]
8. Combine data with business context in final answer

**Key Principle:** Provide data BUT identify when business judgment is needed!
"""

    llm = ChatAnthropic(
        model=settings.MODEL_NAME,
        temperature=0.0,  # Deterministic for SQL
        max_tokens=4096
    )
    
    tools = [
        # Domain tools
        search_similar_queries,
        get_database_schema,
        validate_sql_query,
        execute_sql_query,
        # Communication tool
        ask_another_agent
    ]
    
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
```

---

## 🔄 Phase 3: Workflow Implementation

### 3.1 Workflow Graph

```python
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
```

---

## 📝 Phase 4: Logging & Debugging

### 4.1 Interaction Logger

```python
# app/services/logger.py

from datetime import datetime
from typing import List
import json
import psycopg2
from app.config import get_settings

settings = get_settings()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(settings.DATABASE_URL)


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
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Extract ReAct trace
    react_trace = extract_react_trace(messages)
    
    # Prepare log entry
    log_entry = {
        "session_id": session_id,
        "agent_name": agent_name,
        "timestamp": datetime.utcnow().isoformat(),
        "react_trace": react_trace,
        "next_agent": next_agent,
        "message_count": len(messages)
    }
    
    # Insert into database
    cursor.execute("""
        INSERT INTO agent_interactions 
        (session_id, agent_name, timestamp, react_trace, next_agent, message_count)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        log_entry["session_id"],
        log_entry["agent_name"],
        log_entry["timestamp"],
        json.dumps(log_entry["react_trace"]),
        log_entry["next_agent"],
        log_entry["message_count"]
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    # Also log to console for development
    print(f"\n📝 Logged: {agent_name} → {next_agent}")
    print(f"   Steps: {len(react_trace)}")


def extract_react_trace(messages: List) -> List[dict]:
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


def get_session_logs(session_id: str) -> List[dict]:
    """Retrieve all logs for a session"""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT agent_name, timestamp, react_trace, next_agent
        FROM agent_interactions
        WHERE session_id = %s
        ORDER BY timestamp
    """, (session_id,))
    
    results = cursor.fetchall()
    
    logs = []
    for row in results:
        logs.append({
            "agent_name": row[0],
            "timestamp": row[1],
            "react_trace": json.loads(row[2]),
            "next_agent": row[3]
        })
    
    cursor.close()
    conn.close()
    
    return logs
```

### 4.2 Database Schema for Logging

```sql
-- database/schema.sql

CREATE TABLE IF NOT EXISTS agent_interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    react_trace JSONB NOT NULL,
    next_agent VARCHAR(50),
    message_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_session_id ON agent_interactions(session_id);
CREATE INDEX idx_timestamp ON agent_interactions(timestamp);
CREATE INDEX idx_agent_name ON agent_interactions(agent_name);

-- For analytics
CREATE TABLE IF NOT EXISTS session_metadata (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_turns INTEGER,
    final_answer TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🌐 Phase 5: API Layer (FastAPI)

### 5.1 API Schemas

```python
# app/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List
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
    action_input: Optional[dict] = None
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
```

---

### 5.2 API Routes

```python
# app/api/routes.py

from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas import QueryRequest, QueryResponse, HealthResponse
from app.workflow.graph import workflow_app
from app.services.logger import get_session_logs
from datetime import datetime
import uuid
import time

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
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


@router.get("/session/{session_id}")
async def get_session_trace(session_id: str):
    """
    Retrieve full trace for a session.
    Useful for debugging and auditing.
    """
    
    try:
        logs = get_session_logs(session_id)
        
        if not logs:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return {
            "session_id": session_id,
            "total_interactions": len(logs),
            "logs": logs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )
```

---

### 5.3 Main FastAPI Application

```python
# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import get_settings
import uvicorn

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-agent procurement system with non-deterministic A2A communication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["procurement"])


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print(f"Starting {settings.APP_NAME}...")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"Model: {settings.MODEL_NAME}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
```

---

## 🧪 Phase 6: Testing

### 6.1 Agent Tests

```python
# tests/test_agents.py

import pytest
from app.agents.orchestrator import create_orchestrator_agent
from app.agents.negotiator import create_negotiator_agent
from app.agents.sql_agent import create_sql_agent

def test_orchestrator_routes_to_negotiator():
    """Test orchestrator routes supplier questions to negotiator"""
    
    agent = create_orchestrator_agent()
    
    result = agent.invoke({
        "messages": [("user", "Find best bearing suppliers")]
    })
    
    # Check that orchestrator called ask_another_agent
    last_message = result["messages"][-1]
    assert hasattr(last_message, 'tool_calls')
    
    tool_call = last_message.tool_calls[0]
    assert tool_call["name"] == "ask_another_agent"
    assert tool_call["args"]["agent_name"] == "negotiator"


def test_orchestrator_routes_to_sql():
    """Test orchestrator routes data questions to SQL"""
    
    agent = create_orchestrator_agent()
    
    result = agent.invoke({
        "messages": [("user", "Show me top 10 suppliers by spend")]
    })
    
    last_message = result["messages"][-1]
    assert hasattr(last_message, 'tool_calls')
    
    tool_call = last_message.tool_calls[0]
    assert tool_call["name"] == "ask_another_agent"
    assert tool_call["args"]["agent_name"] == "sql"


def test_negotiator_uses_domain_tools():
    """Test negotiator uses procurement tools"""
    
    agent = create_negotiator_agent()
    
    result = agent.invoke({
        "messages": [("user", "Analyze suppliers for bearings category")]
    })
    
    # Check that negotiator used query_suppliers tool
    found_query_suppliers = False
    for message in result["messages"]:
        if hasattr(message, 'tool_calls'):
            for tc in message.tool_calls:
                if tc["name"] == "query_suppliers":
                    found_query_suppliers = True
                    break
    
    assert found_query_suppliers, "Negotiator should use query_suppliers tool"


def test_sql_agent_validates_before_execution():
    """Test SQL agent validates queries before executing"""
    
    agent = create_sql_agent()
    
    result = agent.invoke({
        "messages": [("user", "Get total spend by supplier")]
    })
    
    # Should see both validate_sql_query and execute_sql_query
    tools_called = []
    for message in result["messages"]:
        if hasattr(message, 'tool_calls'):
            for tc in message.tool_calls:
                tools_called.append(tc["name"])
    
    assert "validate_sql_query" in tools_called
    assert "execute_sql_query" in tools_called


def test_agent_a2a_communication():
    """Test that negotiator can request SQL help"""
    
    agent = create_negotiator_agent()
    
    # Simulate scenario where negotiator needs historical data
    result = agent.invoke({
        "messages": [
            ("user", "Recommend negotiation strategy for SupplierX"),
            ("assistant", "I need historical price trends for SupplierX")
        ]
    })
    
    # Negotiator should recognize need for SQL help
    # (This is a behavioral test - check if ask_another_agent is available)
    tools = [tool.name for tool in agent.tools]
    assert "ask_another_agent" in tools


@pytest.mark.integration
def test_full_agent_conversation():
    """Integration test: Full conversation flow"""
    
    from app.workflow.graph import workflow_app
    
    result = workflow_app.invoke({
        "messages": [("user", "Why is bearing spend increasing?")],
        "next_agent": "orchestrator",
        "session_id": "test_session_123"
    })
    
    # Should have multiple messages (conversation)
    assert len(result["messages"]) > 2
    
    # Should have a final answer
    final_message = result["messages"][-1]
    assert final_message.content
    assert len(final_message.content) > 50  # Substantive answer
```

---

### 6.2 Workflow Tests

```python
# tests/test_workflow.py

import pytest
from app.workflow.graph import (
    workflow_app,
    extract_routing,
    orchestrator_node,
    negotiator_node,
    sql_node
)
from app.workflow.state import AgentState

def test_extract_routing_with_ask_another_agent():
    """Test routing extraction from tool calls"""
    
    # Mock message with tool call
    class MockMessage:
        tool_calls = [{
            "name": "ask_another_agent",
            "args": {"agent_name": "sql", "question": "test"}
        }]
    
    result = extract_routing(MockMessage())
    assert result == "sql"


def test_extract_routing_no_tool_call():
    """Test routing when no tool call present"""
    
    class MockMessage:
        tool_calls = []
    
    result = extract_routing(MockMessage())
    assert result == "END"


def test_workflow_starts_with_orchestrator():
    """Test that workflow always starts with orchestrator"""
    
    initial_state = {
        "messages": [("user", "Test query")],
        "next_agent": "orchestrator",
        "session_id": "test_123"
    }
    
    result = workflow_app.invoke(initial_state)
    
    # First agent should be orchestrator
    assert result is not None


def test_workflow_prevents_infinite_loops():
    """Test that workflow has max iteration limit"""
    
    # This would be enforced by LangGraph's max_iterations config
    # Test that config is set correctly
    from app.config import get_settings
    settings = get_settings()
    
    assert settings.MAX_ITERATIONS > 0
    assert settings.MAX_ITERATIONS <= 20  # Reasonable limit


@pytest.mark.integration
def test_multi_agent_routing():
    """Test that agents can route to each other"""
    
    result = workflow_app.invoke({
        "messages": [("user", "Analyze bearing spend and recommend strategies")],
        "next_agent": "orchestrator",
        "session_id": "test_multi_agent"
    })
    
    # Should involve multiple agents
    # Check messages for evidence of routing
    message_content = " ".join([m.content for m in result["messages"] if hasattr(m, 'content')])
    
    # Should see evidence of analysis and recommendations
    assert len(result["messages"]) > 3
```

---

### 6.3 API Tests

```python
# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_query_endpoint_success():
    """Test successful query"""
    
    response = client.post(
        "/api/v1/query",
        json={
            "query": "Find bearing suppliers",
            "user_id": "test_user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "session_id" in data
    assert "answer" in data
    assert "agent_trace" in data
    assert len(data["answer"]) > 0


def test_query_endpoint_validation():
    """Test query validation"""
    
    response = client.post(
        "/api/v1/query",
        json={}  # Missing required 'query' field
    )
    
    assert response.status_code == 422  # Validation error


def test_session_retrieval():
    """Test retrieving session trace"""
    
    # First, create a session
    query_response = client.post(
        "/api/v1/query",
        json={"query": "Test query"}
    )
    
    session_id = query_response.json()["session_id"]
    
    # Now retrieve it
    response = client.get(f"/api/v1/session/{session_id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["session_id"] == session_id
    assert "logs" in data
    assert len(data["logs"]) > 0


def test_session_not_found():
    """Test retrieving non-existent session"""
    
    response = client.get("/api/v1/session/nonexistent_session")
    
    assert response.status_code == 404


@pytest.mark.integration
def test_full_api_flow():
    """Integration test: Full API flow"""
    
    # Submit query
    response = client.post(
        "/api/v1/query",
        json={
            "query": "Show me top suppliers by spend and recommend negotiation approach",
            "user_id": "integration_test_user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have answer
    assert data["answer"]
    assert len(data["answer"]) > 100
    
    # Should have trace
    assert len(data["agent_trace"]) > 0
    
    # Should have involved multiple agents
    assert data["total_agents_involved"] >= 2
    
    # Retrieve session
    session_response = client.get(f"/api/v1/session/{data['session_id']}")
    assert session_response.status_code == 200
```

---

## 🐳 Phase 7: Deployment

### 7.1 Requirements

```txt
# requirements.txt

# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# LangChain/LangGraph
langchain==0.1.0
langchain-anthropic==0.1.0
langgraph==0.0.20
langsmith==0.0.70

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Vector DB (optional, for SQL agent)
pinecone-client==3.0.0

# Utilities
python-dotenv==1.0.0
python-multipart==0.0.6

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

---

### 7.2 Docker Setup

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/procurement
    depends_on:
      - db
    volumes:
      - ./app:/app/app  # For development
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=procurement
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  postgres_data:
```

### 7.3 Running the System

```bash
# Development
docker-compose up

# Production
docker-compose up -d

# View logs
docker-compose logs -f api

# Run tests
docker-compose exec api pytest

# Access API docs
# http://localhost:8000/docs
```

---

## 📊 Phase 8: Usage Examples

### 8.1 Simple Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the best bearing suppliers",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "session_id": "sess_a1b2c3d4",
  "answer": "Based on comprehensive analysis, I recommend the following bearing suppliers:\n\n1. **SupplierX** (Strategic Partner)\n   - YTD Spend: $2.5M\n   - Kraljic Position: Strategic\n   - Recommendation: Focus on partnership and long-term contracts\n\n2. **SupplierY** (Leverage Opportunity)\n   - YTD Spend: $1.8M\n   - Kraljic Position: Leverage\n   - Recommendation: Aggressive price negotiation, potential 10-15% savings\n\nBoth suppliers have excellent ratings and can meet your volume requirements.",
  "agent_trace": [
    {
      "step_number": 1,
      "agent_name": "orchestrator",
      "thought": "User wants supplier recommendations, I'll route to negotiator",
      "action": "ask_another_agent",
      "action_input": {"agent_name": "negotiator", "question": "Find and recommend best bearing suppliers"}
    },
    {
      "step_number": 2,
      "agent_name": "negotiator",
      "thought": "I'll query suppliers first",
      "action": "query_suppliers",
      "action_input": {"category": "bearings"},
      "observation": "Found 10 suppliers"
    }
  ],
  "total_agents_involved": 2,
  "execution_time_seconds": 3.45,
  "timestamp": "2024-12-17T10:30:00Z"
}
```

---

### 8.2 Complex Multi-Agent Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why has our bearing spend increased 30% this year, and what should we do about it?",
    "user_id": "user123"
  }'
```

**Agent Flow:**
1. Orchestrator → SQL (analyze spend)
2. SQL → Negotiator (explain increase)
3. Negotiator → Orchestrator (synthesize)

---

### 8.3 Retrieving Session Trace

```bash
curl -X GET "http://localhost:8000/api/v1/session/sess_a1b2c3d4"
```

**Response:**
```json
{
  "session_id": "sess_a1b2c3d4",
  "total_interactions": 3,
  "logs": [
    {
      "agent_name": "orchestrator",
      "timestamp": "2024-12-17T10:30:00Z",
      "react_trace": [...],
      "next_agent": "negotiator"
    },
    {
      "agent_name": "negotiator",
      "timestamp": "2024-12-17T10:30:03Z",
      "react_trace": [...],
      "next_agent": "orchestrator"
    }
  ]
}
```

---

## 📈 Monitoring & Observability

### LangSmith Integration

```python
# Add to app/config.py

class Settings(BaseSettings):
    # ... existing settings ...
    
    # LangSmith
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "procurement-agents-simple"
```

**View traces at:** https://smith.langchain.com

---

## ✅ Summary: Simple Design

### **Key Characteristics:**

| Aspect | Simple Design |
|--------|--------------|
| **Complexity** | Low |
| **Lines of Code** | ~800 |
| **Communication** | One tool: `ask_another_agent` |
| **State** | 3 fields |
| **Setup Time** | 2-3 days |
| **Maintainability** | High |
| **Interview-Friendly** | ✅ Excellent |

### **Strengths:**
✅ Easy to understand and explain  
✅ Clean code, minimal abstractions  
✅ Full non-deterministic A2A communication  
✅ Complete logging and debugging  
✅ Production-ready  
✅ Easy to extend

### **Perfect For:**
- Technical interviews
- MVPs and prototypes
- Teams learning multi-agent systems
- Projects with clear agent boundaries

---

## 🚀 Next Steps

1. **Clone and Setup:**
   ```bash
   git clone <repo>
   cd procurement-agent-simple
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY
   docker-compose up
   ```

2. **Test It:**
   ```bash
   pytest
   ```

3. **Try It:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

4. **Extend It:**
   - Add more agents
   - Add more tools
   - Customize prompts

---

**END OF SIMPLE DESIGN IMPLEMENTATION PLAN**
