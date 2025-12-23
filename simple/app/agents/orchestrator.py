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