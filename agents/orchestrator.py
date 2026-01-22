"""
Multi-Agent Orchestrator
Coordinates execution of specialized agents with dependency management
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent, AgentResult
from agents.specialized_agents import (
    EvidenceIngestionAgent,
    MedicalContextAgent,
    CausalInferenceAgent,
    ContradictionResolutionAgent
)
from storage.vault import get_vault
from causal.graph_builder import get_graph_builder
from causal.confidence import get_confidence_scorer, ConfidenceFactors
from causal.trail_extractor import get_trail_extractor
from causal.bias_detector import get_bias_detector, BiasReport
from core.logging_config import get_logger
import time

logger = get_logger("orchestrator")


class AgentOrchestrator:
    """
    Orchestrates multi-agent execution with dependency management.
    Implements the autonomous reasoning loop.
    """
    
    def __init__(self):
        """Initialize orchestrator with agents"""
        self.vault = get_vault()
        
        # Initialize agents
        self.agents = {
            "evidence_ingestion": EvidenceIngestionAgent(),
            "medical_context": MedicalContextAgent(),
            "causal_inference": CausalInferenceAgent(),
            "contradiction_resolution": ContradictionResolutionAgent()
        }
        
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
    
    def execute_pipeline(
        self,
        input_text: str,
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute full reasoning pipeline
        
        Args:
            input_text: Raw input text (medical case, query, etc.)
            source: Source of input
            metadata: Additional metadata
            
        Returns:
            Complete reasoning result with all agent outputs
        """
        start_time = time.time()
        logger.info("Starting reasoning pipeline")
        
        # Log input to vault
        input_id = self.vault.log_input(
            source=source,
            content=input_text,
            metadata=metadata or {}
        )
        
        try:
            # Step 1: Evidence Ingestion
            logger.info("Step 1: Evidence Ingestion")
            evidence_result = self.agents["evidence_ingestion"].execute({
                "text": input_text,
                "source": source,
                "metadata": metadata
            })
            
            # Log to vault
            exec_id_evidence = self.vault.log_agent_execution(
                input_id=input_id,
                agent_id="evidence_ingestion",
                agent_input=input_text,
                agent_output=str(evidence_result.output),
                tool_calls=[],
                duration_ms=evidence_result.duration_ms
            )
            
            extracted_data = evidence_result.output
            
            # Step 2: Medical Context Retrieval (RAG)
            logger.info("Step 2: Medical Context Retrieval")
            context_result = self.agents["medical_context"].execute({
                "extracted_data": extracted_data,
                "top_k": 5
            })
            
            exec_id_context = self.vault.log_agent_execution(
                input_id=input_id,
                agent_id="medical_context",
                agent_input=str(extracted_data),
                agent_output=str(context_result.output),
                duration_ms=context_result.duration_ms
            )
            
            context_documents = context_result.output
            
            # Step 3: Causal Inference
            logger.info("Step 3: Causal Inference")
            causal_result = self.agents["causal_inference"].execute({
                "extracted_data": extracted_data,
                "context_documents": context_documents
            })
            
            exec_id_causal = self.vault.log_agent_execution(
                input_id=input_id,
                agent_id="causal_inference",
                agent_input=str({"extracted_data": extracted_data}),
                agent_output=str(causal_result.output),
                duration_ms=causal_result.duration_ms
            )
            
            # Log causal steps to vault
            causal_chains = causal_result.output.get("causal_chains", [])
            for chain in causal_chains:
                self.vault.log_causal_step(
                    execution_id=exec_id_causal,
                    premise=chain.get("from", ""),
                    conclusion=chain.get("to", ""),
                    confidence=chain.get("confidence", 0.5),
                    evidence_refs=[chain.get("evidence", "")],
                    reasoning_type="causal"
                )
            
            # Step 4: Contradiction Resolution
            logger.info("Step 4: Contradiction Resolution")
            contradiction_result = self.agents["contradiction_resolution"].execute({
                "causal_chains": causal_chains,
                "context_documents": context_documents
            })
            
            exec_id_contradiction = self.vault.log_agent_execution(
                input_id=input_id,
                agent_id="contradiction_resolution",
                agent_input=str({"causal_chains": causal_chains}),
                agent_output=str(contradiction_result.output),
                duration_ms=contradiction_result.duration_ms
            )
            
            # === PHASE 3: Causal Graph & Confidence Scoring ===
            
            # Step 5: Build Causal Graph
            logger.info("Step 5: Building Causal Graph")
            graph_builder = get_graph_builder()
            causal_graph = graph_builder.build_from_results(
                evidence_output=extracted_data,
                context_output=context_documents,
                causal_output=causal_result.output,
                contradiction_output=contradiction_result.output
            )
            
            # Step 6: Calculate Multi-Factor Confidence
            logger.info("Step 6: Calculating Multi-Factor Confidence")
            confidence_scorer = get_confidence_scorer()
            
            # Calculate evidence quality
            evidence_quality = confidence_scorer.calculate_evidence_quality(
                source_credibility=0.7,  # Clinical notes/guidelines
                recency_score=0.9,  # Recent data
                sample_size_score=0.6  # Limited sample
            )
            
            # Calculate reasoning coherence
            has_contradictions = len(contradiction_result.output.get("contradictions", [])) > 0
            is_complete_chain = len(causal_chains) > 0
            reasoning_coherence = confidence_scorer.calculate_reasoning_coherence(
                has_contradictions=has_contradictions,
                is_complete_chain=is_complete_chain,
                logical_consistency_score=0.85
            )
            
            # Aggregate agent confidences for LLM confidence
            agent_confidences = [
                evidence_result.confidence,
                causal_result.confidence,
                contradiction_result.confidence
            ]
            llm_confidence = sum(agent_confidences) / len(agent_confidences)
            
            # Context match from RAG
            context_match = context_result.confidence
            
            # Calculate overall confidence with all factors
            confidence_factors = ConfidenceFactors(
                evidence_quality=evidence_quality,
                reasoning_coherence=reasoning_coherence,
                llm_confidence=llm_confidence,
                context_match=context_match
            )
            overall_confidence = confidence_scorer.calculate_from_factors(confidence_factors)
            confidence_level = confidence_scorer.get_level(overall_confidence)
            
            logger.info(f"Multi-factor confidence: {overall_confidence:.2%} ({confidence_level})")
            
            # Step 7: Detect Bias
            logger.info("Step 7: Check for Bias")
            bias_detector = get_bias_detector()
            bias_report = bias_detector.check_graph(causal_graph, metadata or {})
            
            if bias_report.has_bias:
                logger.warning(f"Potential bias detected: {bias_report.detected_types}")
                risk_flags.append(f"BIAS_DETECTED: {','.join(bias_report.detected_types)}")
            
            # Step 8: Extract Reasoning Trail
            logger.info("Step 8: Extracting Reasoning Trail")
            trail_extractor = get_trail_extractor()
            reasoning_trail = trail_extractor.extract(causal_graph)
            visual_export = trail_extractor.export_graph_for_react_flow(causal_graph)
            trail_summary = trail_extractor.generate_summary(causal_graph, reasoning_trail)
            
            # Generate final conclusion
            conclusion = self._generate_conclusion(
                extracted_data=extracted_data,
                causal_chains=causal_chains,
                contradictions=contradiction_result.output.get("contradictions", []),
                resolutions=contradiction_result.output.get("resolutions", [])
            )
            
            # Determine risk flags
            risk_flags = []
            if overall_confidence < 0.7:
                risk_flags.append("LOW_CONFIDENCE")
            if contradiction_result.output.get("contradictions"):
                risk_flags.append("CONTRADICTIONS_FOUND")
            if not context_documents:
                risk_flags.append("NO_CONTEXT_AVAILABLE")
            
            # Log final output to vault
            output_id = self.vault.log_output(
                execution_id=exec_id_contradiction,
                conclusion=conclusion,
                confidence=overall_confidence,
                risk_flags=risk_flags,
                recommendations=self._generate_recommendations(risk_flags, overall_confidence)
            )
            
            # Compile full result
            total_duration = (time.time() - start_time) * 1000
            
            result = {
                "input_id": input_id,
                "output_id": output_id,
                "conclusion": conclusion,
                "confidence": overall_confidence,
                "confidence_level": confidence_level,
                "confidence_factors": {
                    "evidence_quality": evidence_quality,
                    "reasoning_coherence": reasoning_coherence,
                    "llm_confidence": llm_confidence,
                    "context_match": context_match
                },
                "risk_flags": risk_flags,
                "bias_report": {
                    "has_bias": bias_report.has_bias,
                    "score": bias_report.bias_score,
                    "detected_types": bias_report.detected_types,
                    "recommendations": bias_report.recommendations,
                    "counterfactuals": bias_report.counterfactuals
                },
                "causal_graph": {
                    "graph_id": causal_graph.graph_id,
                    "stats": causal_graph.get_stats(),
                    "export_json": causal_graph.to_json(),
                    "visual_export": visual_export
                },
                "reasoning_trail": reasoning_trail,
                "trail_summary": trail_summary,
                "agent_results": {
                    "evidence_ingestion": {
                        "output": evidence_result.output,
                        "confidence": evidence_result.confidence,
                        "duration_ms": evidence_result.duration_ms
                    },
                    "medical_context": {
                        "output": context_result.output,
                        "confidence": context_result.confidence,
                        "duration_ms": context_result.duration_ms
                    },
                    "causal_inference": {
                        "output": causal_result.output,
                        "confidence": causal_result.confidence,
                        "duration_ms": causal_result.duration_ms
                    },
                    "contradiction_resolution": {
                        "output": contradiction_result.output,
                        "confidence": contradiction_result.confidence,
                        "duration_ms": contradiction_result.duration_ms
                    }
                },
                "total_duration_ms": total_duration
            }
            
            logger.info(
                f"Pipeline complete: {total_duration:.2f}ms, "
                f"confidence: {overall_confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
    
    def _generate_conclusion(
        self,
        extracted_data: Dict[str, Any],
        causal_chains: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        resolutions: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable conclusion"""
        parts = []
        
        # Summary of extracted data
        symptoms = extracted_data.get("symptoms", [])
        diagnoses = extracted_data.get("diagnoses", [])
        treatments = extracted_data.get("treatments", [])
        
        if diagnoses:
            parts.append(f"Identified diagnoses: {', '.join(diagnoses)}")
        
        if symptoms:
            parts.append(f"Based on symptoms: {', '.join(symptoms[:3])}")
        
        # Causal relationships
        if causal_chains:
            parts.append(f"Established {len(causal_chains)} causal relationships")
        
        # Contradictions
        if contradictions:
            parts.append(
                f"Note: {len(contradictions)} contradictions found and "
                f"{len(resolutions)} resolutions proposed"
            )
        
        if not parts:
            parts.append("Insufficient information for conclusive analysis")
        
        return ". ".join(parts) + "."
    
    def _generate_recommendations(
        self,
        risk_flags: List[str],
        confidence: float
    ) -> List[str]:
        """Generate recommendations based on risk flags"""
        recommendations = []
        
        if "LOW_CONFIDENCE" in risk_flags:
            recommendations.append("Human review recommended due to low confidence")
        
        if "CONTRADICTIONS_FOUND" in risk_flags:
            recommendations.append("Expert consultation advised to resolve contradictions")
        
        if "NO_CONTEXT_AVAILABLE" in risk_flags:
            recommendations.append("Consider gathering additional medical context")
        
        if confidence < 0.5:
            recommendations.append("CRITICAL: Confidence below threshold - do not proceed without review")
        
        if not recommendations:
            recommendations.append("Analysis appears sound - proceed with clinical judgment")
        
        return recommendations
    
    def get_reasoning_trail(self, execution_id: int) -> Dict[str, Any]:
        """Retrieve complete reasoning trail from vault"""
        return self.vault.get_reasoning_trail(execution_id)


# Singleton instance
_orchestrator_instance: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create singleton Orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance
