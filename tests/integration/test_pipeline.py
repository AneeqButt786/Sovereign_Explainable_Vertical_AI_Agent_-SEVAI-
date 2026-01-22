"""
Integration Tests for SEVAI Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.orchestrator import AgentOrchestrator
from storage.vault import ExplainabilityVault

class TestPipelineIntegration:
    @patch('core.llm_manager.OpenAI')
    @patch('storage.vector_store.Pinecone')
    @patch('agents.orchestrator.get_vault')
    def test_full_flow(self, mock_get_vault, mock_pinecone, mock_openai):
        """
        Test end-to-end flow from input to output with mocked external APIs.
        Real agents, real logic, but fake LLM/Pinecone responses.
        Uses real SQLite in-memory for auditing.
        """
        # Setup real in-memory vault for testing
        test_vault = ExplainabilityVault("sqlite:///:memory:")
        mock_get_vault.return_value = test_vault

        # Setup OpenAI mock to return valid JSON for agents
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        
        # We need simpler returns to avoid complex side_effects for every call
        # But agents expect specific JSON structures
        mock_choice.message.content = '{"symptoms": ["fever"], "diagnoses": ["flu"], "causal_chains": [], "contradictions": []}' 
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client
        
        # Setup Pinecone mock
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {'total_vector_count': 0}
        mock_pinecone.return_value.Index.return_value = mock_index
        
        # Run pipeline
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute_pipeline("Patient has fever")
        
        assert result["confidence"] > 0
        assert "agent_results" in result
        assert "causal_graph" in result
        
        # Verify it actually logged something to our in-memory DB
        with test_vault.SessionLocal() as session:
            from storage.vault import Input
            inputs = session.query(Input).all()
            assert len(inputs) > 0
            assert inputs[0].content == "Patient has fever"
