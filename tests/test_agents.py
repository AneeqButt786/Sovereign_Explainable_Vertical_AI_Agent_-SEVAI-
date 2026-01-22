"""
Tests for Phase 2 core components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.base_agent import BaseAgent, AgentResult
from agents.specialized_agents import EvidenceIngestionAgent


class SimpleTestAgent(BaseAgent):
    """Simple agent for testing"""
    
    def ingest(self, input_data):
        return {"processed": input_data}
    
    def reason(self, processed_input):
        return {
            "result": "test_result",
            "confidence": 0.9,
            "conclusions": ["test conclusion"]
        }
    
    def output(self, reasoning_result):
        return reasoning_result["result"]


def test_base_agent_execute():
    """Test base agent execution pipeline"""
    agent = SimpleTestAgent("test_agent", "Test agent")
    
    result = agent.execute({"input": "test"})
    
    assert isinstance(result, AgentResult)
    assert result.agent_id == "test_agent"
    assert result.output == "test_result"
    assert result.confidence == 0.9
    assert len(result.reasoning_steps) == 3  # ingest, reason, output


@patch('agents.specialized_agents.get_llm_manager')
def test_evidence_ingestion_agent(mock_llm):
    """Test evidence ingestion agent"""
    # Mock LLM response
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = {
        "output": '{"symptoms": ["fever", "cough"], "diagnoses": ["flu"], "treatments": [], "medications": [], "test_results": [], "outcomes": []}',
        "cost": 0.001
    }
    mock_llm.return_value = mock_llm_instance
    
    agent = EvidenceIngestionAgent()
    result = agent.execute({
        "text": "Patient has fever and cough, diagnosed with flu."
    })
    
    assert isinstance(result, AgentResult)
    assert "symptoms" in result.output
    assert "fever" in result.output["symptoms"]


def test_base_agent_repr():
    """Test agent string representation"""
    agent = SimpleTestAgent("test_agent")
    assert "SimpleTestAgent" in str(agent)
    assert "test_agent" in str(agent)
