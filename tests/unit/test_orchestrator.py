"""
Tests for Agent Orchestrator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.orchestrator import AgentOrchestrator
from agents.base_agent import AgentResult

class TestOrchestrator:
    @pytest.fixture
    def mock_agents(self):
        with patch('agents.orchestrator.EvidenceIngestionAgent') as mock_ev, \
             patch('agents.orchestrator.MedicalContextAgent') as mock_mc, \
             patch('agents.orchestrator.CausalInferenceAgent') as mock_ci, \
             patch('agents.orchestrator.ContradictionResolutionAgent') as mock_cr, \
             patch('agents.orchestrator.get_vault') as mock_vault, \
             patch('agents.orchestrator.get_graph_builder') as mock_gb, \
             patch('agents.orchestrator.get_confidence_scorer') as mock_cs, \
             patch('agents.orchestrator.get_trail_extractor') as mock_te, \
             patch('agents.orchestrator.get_bias_detector') as mock_bd:
            
            # Setup agent instances
            common_result_args = {
                "reasoning_steps": [],
                "tool_calls": [],
                "metadata": {},
                "timestamp": None
            }
            
            ev_inst = MagicMock()
            ev_inst.execute.return_value = AgentResult(agent_id="evidence", output={"symptoms": []}, confidence=0.9, duration_ms=10, **common_result_args)
            mock_ev.return_value = ev_inst
            
            mc_inst = MagicMock()
            mc_inst.execute.return_value = AgentResult(agent_id="context", output=[], confidence=0.8, duration_ms=10, **common_result_args)
            mock_mc.return_value = mc_inst
            
            ci_inst = MagicMock()
            ci_inst.execute.return_value = AgentResult(agent_id="causal", output={"causal_chains": []}, confidence=0.85, duration_ms=10, **common_result_args)
            mock_ci.return_value = ci_inst
            
            cr_inst = MagicMock()
            cr_inst.execute.return_value = AgentResult(agent_id="resolution", output={"contradictions": []}, confidence=0.9, duration_ms=10, **common_result_args)
            mock_cr.return_value = cr_inst
            
            # Setup helpers
            mock_gb.return_value.build_from_results.return_value = MagicMock(graph_id="test_graph")
            mock_cs.return_value.calculate_from_factors.return_value = 0.88
            mock_cs.return_value.get_level.return_value = "high"
            mock_te.return_value.extract.return_value = {"paths": []}
            
            bd_report = MagicMock()
            bd_report.has_bias = False
            bd_report.detected_types = []
            mock_bd.return_value.check_graph.return_value = bd_report
            
            yield {
                "ev": ev_inst,
                "mc": mc_inst, 
                "ci": ci_inst,
                "cr": cr_inst,
                "vault": mock_vault
            }

    def test_pipeline_execution(self, mock_agents):
        """Test full pipeline execution flow"""
        orchestrator = AgentOrchestrator()
        
        result = orchestrator.execute_pipeline("Patient has fever")
        
        # Verify agents called in order
        assert mock_agents["ev"].execute.called
        assert mock_agents["mc"].execute.called
        assert mock_agents["ci"].execute.called
        assert mock_agents["cr"].execute.called
        
        # Verify result structure
        assert "input_id" in result
        assert "output_id" in result
        assert "confidence" in result
        assert "causal_graph" in result
        assert "bias_report" in result
        
        # Verify vault logging
        assert mock_agents["vault"].return_value.log_input.called
        assert mock_agents["vault"].return_value.log_output.called
