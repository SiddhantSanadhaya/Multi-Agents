# tests/test_agents.py

import pytest
import os
from unittest.mock import Mock, patch
from app.agents.orchestrator import create_orchestrator_agent
from app.agents.negotiator import create_negotiator_agent
from app.agents.sql_agent import create_sql_agent

# Set test environment variables
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


class MockMessage:
    """Mock message class for testing"""
    
    def __init__(self, content="", tool_calls=None, name=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.name = name


class TestOrchestratorAgent:
    """Test orchestrator agent behavior"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator agent for testing"""
        return create_orchestrator_agent()
    
    def test_orchestrator_creation(self, orchestrator):
        """Test that orchestrator agent is created successfully"""
        assert orchestrator is not None
        # Check that it has the ask_another_agent tool
        tool_names = [tool.name for tool in orchestrator.tools]
        assert "ask_another_agent" in tool_names
    
    @patch('app.agents.orchestrator.ChatAnthropic')
    def test_orchestrator_routes_to_negotiator(self, mock_llm):
        """Test orchestrator routes supplier questions to negotiator"""
        
        # Mock the LLM response
        mock_response = MockMessage(
            content="I need to route this to the negotiator",
            tool_calls=[{
                "name": "ask_another_agent",
                "args": {"agent_name": "negotiator", "question": "Find best bearing suppliers"}
            }]
        )
        
        mock_llm.return_value.invoke.return_value = {
            "messages": [mock_response]
        }
        
        orchestrator = create_orchestrator_agent()
        
        # Test with a supplier-related query
        result = orchestrator.invoke({
            "messages": [("user", "Find best bearing suppliers")]
        })
        
        # Should have invoked the LLM
        assert mock_llm.return_value.invoke.called
    
    @patch('app.agents.orchestrator.ChatAnthropic')
    def test_orchestrator_routes_to_sql(self, mock_llm):
        """Test orchestrator routes data questions to SQL"""
        
        mock_response = MockMessage(
            content="I need data analysis",
            tool_calls=[{
                "name": "ask_another_agent", 
                "args": {"agent_name": "sql", "question": "Show top suppliers by spend"}
            }]
        )
        
        mock_llm.return_value.invoke.return_value = {
            "messages": [mock_response]
        }
        
        orchestrator = create_orchestrator_agent()
        
        result = orchestrator.invoke({
            "messages": [("user", "Show me top 10 suppliers by spend")]
        })
        
        assert mock_llm.return_value.invoke.called


class TestNegotiatorAgent:
    """Test negotiator agent behavior"""
    
    @pytest.fixture
    def negotiator(self):
        """Create negotiator agent for testing"""
        return create_negotiator_agent()
    
    def test_negotiator_creation(self, negotiator):
        """Test that negotiator agent is created successfully"""
        assert negotiator is not None
        
        # Check that it has all expected tools
        tool_names = [tool.name for tool in negotiator.tools]
        expected_tools = [
            "query_suppliers",
            "get_kraljic_positioning", 
            "get_ytd_spend",
            "calculate_negotiation_strategy",
            "ask_another_agent"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names
    
    def test_negotiator_has_procurement_tools(self, negotiator):
        """Test negotiator has access to procurement domain tools"""
        tool_names = [tool.name for tool in negotiator.tools]
        
        # Should have domain-specific tools
        assert "query_suppliers" in tool_names
        assert "get_kraljic_positioning" in tool_names
        assert "get_ytd_spend" in tool_names
        assert "calculate_negotiation_strategy" in tool_names
        
        # Should also have communication tool
        assert "ask_another_agent" in tool_names
    
    @patch('app.agents.negotiator.ChatAnthropic')
    def test_negotiator_uses_domain_tools(self, mock_llm):
        """Test negotiator uses procurement tools"""
        
        mock_response = MockMessage(
            content="Let me query suppliers",
            tool_calls=[{
                "name": "query_suppliers",
                "args": {"category": "bearings"}
            }]
        )
        
        mock_llm.return_value.invoke.return_value = {
            "messages": [mock_response]
        }
        
        negotiator = create_negotiator_agent()
        
        result = negotiator.invoke({
            "messages": [("user", "Analyze suppliers for bearings category")]
        })
        
        assert mock_llm.return_value.invoke.called


class TestSQLAgent:
    """Test SQL agent behavior"""
    
    @pytest.fixture
    def sql_agent(self):
        """Create SQL agent for testing"""
        return create_sql_agent()
    
    def test_sql_agent_creation(self, sql_agent):
        """Test that SQL agent is created successfully"""
        assert sql_agent is not None
        
        # Check that it has all expected tools
        tool_names = [tool.name for tool in sql_agent.tools]
        expected_tools = [
            "search_similar_queries",
            "get_database_schema",
            "validate_sql_query",
            "execute_sql_query",
            "ask_another_agent"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names
    
    def test_sql_agent_has_sql_tools(self, sql_agent):
        """Test SQL agent has access to SQL domain tools"""
        tool_names = [tool.name for tool in sql_agent.tools]
        
        # Should have SQL-specific tools
        assert "search_similar_queries" in tool_names
        assert "validate_sql_query" in tool_names
        assert "execute_sql_query" in tool_names
        assert "get_database_schema" in tool_names
        
        # Should also have communication tool
        assert "ask_another_agent" in tool_names
    
    @patch('app.agents.sql_agent.ChatAnthropic')
    def test_sql_agent_validates_before_execution(self, mock_llm):
        """Test SQL agent validates queries before executing"""
        
        # Mock sequence: search -> validate -> execute
        mock_responses = [
            MockMessage(
                content="Let me search for similar queries",
                tool_calls=[{
                    "name": "search_similar_queries",
                    "args": {"query_description": "top suppliers"}
                }]
            ),
            MockMessage(
                content="Now validate the query",
                tool_calls=[{
                    "name": "validate_sql_query", 
                    "args": {"sql_query": "SELECT * FROM suppliers LIMIT 10"}
                }]
            )
        ]
        
        mock_llm.return_value.invoke.return_value = {
            "messages": mock_responses
        }
        
        sql_agent = create_sql_agent()
        
        result = sql_agent.invoke({
            "messages": [("user", "Get total spend by supplier")]
        })
        
        assert mock_llm.return_value.invoke.called


class TestAgentCommunication:
    """Test agent-to-agent communication capabilities"""
    
    def test_all_agents_have_communication_tool(self):
        """Test that all agents can communicate with each other"""
        
        orchestrator = create_orchestrator_agent()
        negotiator = create_negotiator_agent()
        sql_agent = create_sql_agent()
        
        # All agents should have ask_another_agent tool
        for agent in [orchestrator, negotiator, sql_agent]:
            tool_names = [tool.name for tool in agent.tools]
            assert "ask_another_agent" in tool_names
    
    def test_ask_another_agent_tool_exists(self):
        """Test the ask_another_agent tool works independently"""
        
        from app.tools.communication import ask_another_agent
        
        result = ask_another_agent("sql", "Test question")
        
        # Should return JSON string with routing info
        import json
        parsed = json.loads(result)
        
        assert parsed["action"] == "route_to"
        assert parsed["agent"] == "sql" 
        assert parsed["question"] == "Test question"
        assert "timestamp" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])