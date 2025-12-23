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