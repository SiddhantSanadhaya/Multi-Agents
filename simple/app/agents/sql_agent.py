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