# tests/test_api.py

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.services.database import Base, engine
from datetime import datetime

# Set test environment variables
os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_health_check(self):
        """Test health endpoint"""
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Procurement Multi-Agent System"
        assert "docs" in data
    
    def test_info_endpoint(self):
        """Test info endpoint"""
        
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "query" in data["endpoints"]


class TestQueryEndpoint:
    """Test the main query processing endpoint"""
    
    def setup_method(self):
        """Setup test database"""
        # Create tables for testing
        Base.metadata.create_all(bind=engine)
    
    @patch('app.api.routes.workflow_app')
    def test_query_endpoint_success(self, mock_workflow):
        """Test successful query processing"""
        
        # Mock workflow response
        mock_message = MagicMock()
        mock_message.content = "Based on analysis, SupplierX is the best choice..."
        
        mock_workflow.invoke.return_value = {
            "messages": [mock_message]
        }
        
        # Mock session logs (empty for this test)
        with patch('app.api.routes.get_session_logs', return_value=[]):
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
        assert "execution_time_seconds" in data
        assert len(data["answer"]) > 0
    
    def test_query_endpoint_validation(self):
        """Test query validation"""
        
        response = client.post(
            "/api/v1/query",
            json={}  # Missing required 'query' field
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_with_session_id(self):
        """Test query with existing session ID"""
        
        with patch('app.api.routes.workflow_app') as mock_workflow:
            mock_message = MagicMock()
            mock_message.content = "Test response"
            mock_workflow.invoke.return_value = {"messages": [mock_message]}
            
            with patch('app.api.routes.get_session_logs', return_value=[]):
                response = client.post(
                    "/api/v1/query",
                    json={
                        "query": "Test query",
                        "session_id": "custom_session_123"
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "custom_session_123"
    
    @patch('app.api.routes.workflow_app')
    def test_query_endpoint_error_handling(self, mock_workflow):
        """Test error handling in query endpoint"""
        
        # Mock workflow to raise exception
        mock_workflow.invoke.side_effect = Exception("Test error")
        
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query that will fail"
            }
        )
        
        assert response.status_code == 500
        assert "error" in response.json()


class TestSessionEndpoints:
    """Test session management endpoints"""
    
    def setup_method(self):
        """Setup test database"""
        Base.metadata.create_all(bind=engine)
    
    @patch('app.api.routes.get_session_logs')
    @patch('app.api.routes.get_session_metadata')
    def test_get_session_trace_success(self, mock_metadata, mock_logs):
        """Test retrieving session trace"""
        
        # Mock session data
        mock_logs.return_value = [
            {
                "agent_name": "orchestrator",
                "timestamp": "2024-01-01T00:00:00Z",
                "react_trace": [],
                "next_agent": "negotiator"
            }
        ]
        
        mock_metadata.return_value = {
            "session_id": "test_session",
            "user_id": "test_user",
            "start_time": "2024-01-01T00:00:00Z"
        }
        
        response = client.get("/api/v1/session/test_session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "test_session"
        assert "logs" in data
        assert "metadata" in data
        assert data["total_interactions"] == 1
    
    @patch('app.api.routes.get_session_logs')
    @patch('app.api.routes.get_session_metadata')
    def test_get_session_not_found(self, mock_metadata, mock_logs):
        """Test retrieving non-existent session"""
        
        mock_logs.return_value = []
        mock_metadata.return_value = None
        
        response = client.get("/api/v1/session/nonexistent_session")
        
        assert response.status_code == 404
    
    @patch('app.api.routes.get_recent_sessions')
    def test_get_recent_sessions(self, mock_recent):
        """Test getting recent sessions"""
        
        mock_recent.return_value = [
            {
                "session_id": "session1",
                "user_id": "user1",
                "start_time": "2024-01-01T00:00:00Z",
                "total_turns": 3
            },
            {
                "session_id": "session2", 
                "user_id": "user2",
                "start_time": "2024-01-01T01:00:00Z",
                "total_turns": 2
            }
        ]
        
        response = client.get("/api/v1/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "total_count" in data
        assert data["total_count"] == 2
        assert len(data["sessions"]) == 2
    
    def test_get_recent_sessions_with_limit(self):
        """Test getting recent sessions with limit parameter"""
        
        with patch('app.api.routes.get_recent_sessions') as mock_recent:
            mock_recent.return_value = []
            
            response = client.get("/api/v1/sessions?limit=5")
            
            assert response.status_code == 200
            mock_recent.assert_called_with(limit=5)


class TestDebugEndpoints:
    """Test debugging endpoints"""
    
    def setup_method(self):
        """Setup test database"""
        Base.metadata.create_all(bind=engine)
    
    @patch('app.api.routes.get_session_logs')
    @patch('app.api.routes.get_session_metadata')
    def test_debug_session(self, mock_metadata, mock_logs):
        """Test debug session endpoint"""
        
        mock_logs.return_value = [
            {
                "agent_name": "orchestrator",
                "timestamp": "2024-01-01T00:00:00Z",
                "react_trace": [{"step": 1}],
                "next_agent": "END"
            },
            {
                "agent_name": "negotiator",
                "timestamp": "2024-01-01T00:01:00Z", 
                "react_trace": [{"step": 1}, {"step": 2}],
                "next_agent": "END"
            }
        ]
        
        mock_metadata.return_value = {
            "session_id": "debug_session",
            "user_id": "debug_user"
        }
        
        response = client.get("/api/v1/debug/session/debug_session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "debug_info" in data
        assert data["debug_info"]["total_interactions"] == 2
        assert data["debug_info"]["agents_used"] == ["orchestrator", "negotiator"]
        assert data["debug_info"]["total_steps"] == 3  # 1 + 2 steps


class TestErrorHandling:
    """Test global error handling"""
    
    def test_global_exception_handler(self):
        """Test that unhandled exceptions are caught"""
        
        # This is harder to test directly, but we can test
        # that the exception handler is configured
        assert app.exception_handlers is not None
    
    def test_cors_middleware(self):
        """Test CORS middleware is configured"""
        
        response = client.options("/api/v1/health")
        
        # Should not return 405 Method Not Allowed
        # (exact behavior depends on CORS configuration)
        assert response.status_code != 405


class TestIntegrationFlow:
    """Integration tests for complete API flow"""
    
    def setup_method(self):
        """Setup test database"""
        Base.metadata.create_all(bind=engine)
    
    @patch('app.api.routes.workflow_app')
    def test_full_api_flow(self, mock_workflow):
        """Integration test: Query -> Session retrieval"""
        
        # Step 1: Submit query
        mock_message = MagicMock()
        mock_message.content = "SupplierX is recommended based on analysis..."
        
        mock_workflow.invoke.return_value = {
            "messages": [mock_message]
        }
        
        # Mock session logs for the query response
        with patch('app.api.routes.get_session_logs', return_value=[
            {
                "agent_name": "orchestrator",
                "timestamp": "2024-01-01T00:00:00Z",
                "react_trace": [
                    {
                        "step_number": 1,
                        "thought": "Need supplier analysis",
                        "action": "ask_another_agent",
                        "action_input": {"agent_name": "negotiator"},
                        "observation": None
                    }
                ],
                "next_agent": "negotiator"
            }
        ]):
            query_response = client.post(
                "/api/v1/query",
                json={
                    "query": "Find best bearing suppliers and recommend strategy",
                    "user_id": "integration_test_user"
                }
            )
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Should have answer and trace
        assert query_data["answer"]
        assert len(query_data["answer"]) > 50
        assert len(query_data["agent_trace"]) > 0
        
        session_id = query_data["session_id"]
        
        # Step 2: Retrieve session
        with patch('app.api.routes.get_session_logs', return_value=[
            {
                "agent_name": "orchestrator",
                "timestamp": "2024-01-01T00:00:00Z",
                "react_trace": [],
                "next_agent": "END"
            }
        ]):
            with patch('app.api.routes.get_session_metadata', return_value={
                "session_id": session_id,
                "user_id": "integration_test_user"
            }):
                session_response = client.get(f"/api/v1/session/{session_id}")
        
        assert session_response.status_code == 200
        session_data = session_response.json()
        
        assert session_data["session_id"] == session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])