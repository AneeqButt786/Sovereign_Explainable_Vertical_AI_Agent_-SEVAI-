"""
Tests for Vault storage
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from storage.vault import ExplainabilityVault as Vault, Input, AgentExecution, Output as AgentOutput
from datetime import datetime
import json

class TestVault:
    @pytest.fixture
    def mock_session_instance(self):
        """The actual session object that does work"""
        return MagicMock()

    @pytest.fixture
    def mock_session_factory(self, mock_session_instance):
        """The sessionmaker that returns the session object"""
        factory = MagicMock(return_value=mock_session_instance)
        return factory
        
    @pytest.fixture
    def vault(self, mock_session_factory):
        with patch('storage.vault.create_engine'), \
             patch('storage.vault.sessionmaker', return_value=mock_session_factory):
            vault = Vault("sqlite:///:memory:")
            # We don't need to manually set vault.session here because Vault calls SessionLocal()
            return vault

    def test_log_input(self, vault, mock_session_instance):
        """Test logging user input"""
        # Setup mock returns
        mock_input = MagicMock()
        mock_input.id = 1
        mock_session_instance.add.return_value = None
        
        # Test
        input_id = vault.log_input("user", "test content", {"meta": "data"})
        
        # Verify
        assert mock_session_instance.add.called
        # Verify hash generation logic is implicitly tested by successful add
        args = mock_session_instance.add.call_args[0][0]
        assert isinstance(args, Input)
        assert args.source == "user"
        assert args.content == "test content"
        assert args.content_hash is not None

    def test_log_agent_execution(self, vault, mock_session_instance):
        """Test logging agent execution"""
        mock_exec = MagicMock()
        mock_exec.id = 2
        
        exec_id = vault.log_agent_execution(
            agent_id="test_agent",
            agent_input={"in": "data"},
            agent_output={"out": "data"},
            duration_ms=100.0,
            input_id=1
        )
        
        assert mock_session_instance.add.called
        args = mock_session_instance.add.call_args[0][0]
        assert isinstance(args, AgentExecution)
        assert args.agent_id == "test_agent"
        assert args.duration_ms == 100.0

    def test_get_reasoning_trail(self, vault, mock_session_instance):
        """Test retrieving reasoning trail"""
        # Setup complex mock chain
        mock_output = MagicMock(spec=AgentOutput)
        mock_output.conclusion = "Fatal error"
        mock_output.confidence = 0.9
        
        mock_exec = MagicMock(spec=AgentExecution)
        mock_exec.id = 5
        mock_exec.agent_id = "final_agent"
        mock_exec.input_id = 1
        
        mock_input = MagicMock(spec=Input)
        mock_input.id = 1
        mock_input.content = "Patient data"
        
        # Mock queries
        # The chain is session.query(Class).filter_by(...).first/all
        # We need to mock the FIRST call to query to return a mock query object
        mock_query = mock_session_instance.query.return_value
        mock_filter = mock_query.filter.return_value  # Vault uses filter() not filter_by() in some places, but check code
        # Actually Vault uses filter_by mostly for simple gets, let's check source.
        # Source uses filter_by: query(AgentExecution).filter_by(id=execution_id)
        
        # When filter_by is called, it returns the query object again (fluent API)
        mock_session_instance.query.return_value.filter_by.return_value.first.side_effect = [
            mock_exec,   # 1. Get AgentExecution
            mock_output  # 3. Get Output (Wait, order in code: Exec, Causal, Policy, Output)
        ]
        
        # Order in code:
        # 1. query(AgentExecution).filter_by(id=...).first()
        # 2. query(CausalStep).filter_by(execution_id=...).all()
        # 3. query(PolicyCheck).filter_by(execution_id=...).all()
        # 4. query(Output).filter_by(execution_id=...).first()
        
        # We need to be careful with independent query calls
        # Simplest is to map based on call arg, but side_effect on ONE query mock is hard if called with diff classes
        
        def query_side_effect(model_class):
            m_q = MagicMock()
            if model_class == AgentExecution:
                m_q.filter_by.return_value.first.return_value = mock_exec
            elif model_class == AgentOutput: # Output class
                m_q.filter_by.return_value.first.return_value = mock_output
            elif model_class == Input: # Not used in get_reasoning_trail currently based on viewed code
                pass
            return m_q
            
        mock_session_instance.query.side_effect = query_side_effect
        
        result = vault.get_reasoning_trail(execution_id=10)
        
        assert result is not None
        assert result["output"]["conclusion"] == "Fatal error"
