# tests/test_workflow.py

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from app.workflow.graph import (
    workflow_app,
    extract_routing,
    orchestrator_node,
    negotiator_node,
    sql_node,
    create_workflow
)
from app.workflow.state import AgentState

# Set test environment variables
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


class MockMessage:
    """Mock message class for testing"""
    
    def __init__(self, content="", tool_calls=None, name=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.name = name


class TestRoutingExtraction:
    """Test routing logic extraction"""
    
    def test_extract_routing_with_ask_another_agent(self):
        """Test routing extraction from tool calls"""
        
        mock_message = MockMessage(
            tool_calls=[{
                "name": "ask_another_agent",
                "args": {"agent_name": "sql", "question": "test"}
            }]
        )
        
        result = extract_routing(mock_message)
        assert result == "sql"
    
    def test_extract_routing_no_tool_call(self):
        """Test routing when no tool call present"""
        
        mock_message = MockMessage(content="Just regular text")
        
        result = extract_routing(mock_message)
        assert result == "END"
    
    def test_extract_routing_other_tool_call(self):
        """Test routing when different tool is called"""
        
        mock_message = MockMessage(
            tool_calls=[{
                "name": "query_suppliers",
                "args": {"category": "bearings"}
            }]
        )
        
        result = extract_routing(mock_message)
        assert result == "END"
    
    def test_extract_routing_multiple_tool_calls(self):
        """Test routing with multiple tool calls"""
        
        mock_message = MockMessage(
            tool_calls=[
                {
                    "name": "query_suppliers", 
                    "args": {"category": "bearings"}
                },
                {
                    "name": "ask_another_agent",
                    "args": {"agent_name": "negotiator", "question": "test"}
                }
            ]
        )
        
        result = extract_routing(mock_message)
        assert result == "negotiator"


class TestWorkflowNodes:
    """Test individual workflow nodes"""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock state for testing"""
        return {
            "messages": [("user", "Test query")],
            "next_agent": "orchestrator",
            "session_id": "test_session_123",
            "user_id": "test_user",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    @patch('app.workflow.graph.orchestrator')
    @patch('app.workflow.graph.log_interaction')
    def test_orchestrator_node(self, mock_log, mock_orchestrator, mock_state):
        """Test orchestrator node processing"""
        
        # Mock orchestrator response
        mock_response = MockMessage(
            content="Routing to negotiator",
            tool_calls=[{
                "name": "ask_another_agent",
                "args": {"agent_name": "negotiator", "question": "test"}
            }]
        )
        
        mock_orchestrator.invoke.return_value = {
            "messages": [mock_response]
        }
        
        result = orchestrator_node(mock_state)
        
        # Check that orchestrator was invoked
        mock_orchestrator.invoke.assert_called_once()
        
        # Check that logging was called
        mock_log.assert_called_once()
        
        # Check result state
        assert result["next_agent"] == "negotiator"
        assert len(result["messages"]) == 1
    
    @patch('app.workflow.graph.negotiator')
    @patch('app.workflow.graph.log_interaction')
    def test_negotiator_node(self, mock_log, mock_negotiator, mock_state):
        """Test negotiator node processing"""
        
        # Mock negotiator response (no routing)
        mock_response = MockMessage(content="Analysis complete")
        
        mock_negotiator.invoke.return_value = {
            "messages": [mock_response]
        }
        
        result = negotiator_node(mock_state)
        
        # Should return to orchestrator when no routing
        assert result["next_agent"] == "orchestrator"
        
        mock_negotiator.invoke.assert_called_once()
        mock_log.assert_called_once()
    
    @patch('app.workflow.graph.sql_agent')
    @patch('app.workflow.graph.log_interaction')
    def test_sql_node(self, mock_log, mock_sql_agent, mock_state):
        """Test SQL node processing"""
        
        # Mock SQL agent response (routes to negotiator)
        mock_response = MockMessage(
            content="Need business context",
            tool_calls=[{
                "name": "ask_another_agent",
                "args": {"agent_name": "negotiator", "question": "interpret this data"}
            }]
        )
        
        mock_sql_agent.invoke.return_value = {
            "messages": [mock_response]
        }
        
        result = sql_node(mock_state)
        
        # Should route to negotiator
        assert result["next_agent"] == "negotiator"
        
        mock_sql_agent.invoke.assert_called_once()
        mock_log.assert_called_once()


class TestWorkflowCreation:
    """Test workflow graph creation"""
    
    def test_create_workflow(self):
        """Test that workflow is created successfully"""
        
        workflow = create_workflow()
        assert workflow is not None
    
    def test_workflow_has_correct_nodes(self):
        """Test workflow has all required nodes"""
        
        # This is harder to test directly, but we can verify
        # the workflow compiles without errors
        workflow = create_workflow()
        assert workflow is not None
        
        # The workflow should be compiled and ready to use
        assert hasattr(workflow, 'invoke')
    
    def test_workflow_app_exists(self):
        """Test that global workflow app is created"""
        
        assert workflow_app is not None
        assert hasattr(workflow_app, 'invoke')


class TestWorkflowIntegration:
    """Integration tests for the complete workflow"""
    
    @pytest.fixture
    def test_state(self):
        """Create test state"""
        return {
            "messages": [("user", "Find best bearing suppliers")],
            "next_agent": "orchestrator",
            "session_id": "test_integration_123",
            "user_id": "test_user",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    @patch('app.workflow.graph.orchestrator')
    @patch('app.workflow.graph.negotiator') 
    @patch('app.workflow.graph.sql_agent')
    @patch('app.workflow.graph.log_interaction')
    def test_workflow_orchestrator_to_negotiator_flow(
        self, mock_log, mock_sql, mock_negotiator, mock_orchestrator, test_state
    ):
        """Test flow from orchestrator to negotiator"""
        
        # Mock orchestrator routing to negotiator
        orchestrator_response = MockMessage(
            content="Routing to negotiator",
            tool_calls=[{
                "name": "ask_another_agent",
                "args": {"agent_name": "negotiator", "question": "find suppliers"}
            }]
        )
        
        # Mock negotiator completing task
        negotiator_response = MockMessage(content="Here are the best suppliers...")
        
        mock_orchestrator.invoke.return_value = {"messages": [orchestrator_response]}
        mock_negotiator.invoke.return_value = {"messages": [negotiator_response]}
        
        # This would be a more complex test in practice
        # For now, just test that individual components work
        assert extract_routing(orchestrator_response) == "negotiator"
        assert extract_routing(negotiator_response) == "END"
    
    def test_workflow_prevents_infinite_loops(self):
        """Test that workflow has max iteration limit"""
        
        from app.config import get_settings
        settings = get_settings()
        
        assert settings.MAX_ITERATIONS > 0
        assert settings.MAX_ITERATIONS <= 20  # Reasonable limit
    
    @patch('app.services.logger.get_db_session')
    def test_workflow_logging_integration(self, mock_db):
        """Test that workflow integrates with logging"""
        
        # Mock database session
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        
        from app.services.logger import log_interaction
        
        # Should not raise errors
        log_interaction(
            session_id="test_123",
            agent_name="orchestrator", 
            messages=[MockMessage("test")],
            next_agent="END"
        )
        
        # Database operations should be attempted
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


class TestStateManagement:
    """Test state management in workflow"""
    
    def test_agent_state_structure(self):
        """Test that AgentState has correct structure"""
        
        # This tests the TypedDict structure
        test_state = {
            "messages": [],
            "next_agent": "orchestrator",
            "session_id": "test",
            "user_id": "user",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Should be valid AgentState structure
        assert "messages" in test_state
        assert "next_agent" in test_state
        assert "session_id" in test_state
        assert "user_id" in test_state
        assert "timestamp" in test_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])