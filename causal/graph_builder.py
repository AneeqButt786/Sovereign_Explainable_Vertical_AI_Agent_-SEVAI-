"""
Graph Builder - Automatic Causal Graph Construction
Builds causal graphs from agent reasoning outputs
"""

from typing import Dict, Any, List, Optional
from causal.graph import CausalGraph, NodeType, EdgeType, CausalStrength
from causal.confidence import get_confidence_scorer
from core.logging_config import get_logger

logger = get_logger("graph_builder")


class GraphBuilder:
    """
    Automatically constructs causal graphs from agent outputs.
    Parses reasoning results and builds structured graph representation.
    """
    
    def __init__(self):
        """Initialize graph builder"""
        self.confidence_scorer = get_confidence_scorer()
        logger.info("Graph builder initialized")
    
    def build_from_results(
        self,
        evidence_output: Dict[str, Any],
        context_output: List[Dict[str, Any]],
        causal_output: Dict[str, Any],
        contradiction_output: Dict[str, Any]
    ) -> CausalGraph:
        """
        Build causal graph from agent outputs
        
        Args:
            evidence_output: Output from Evidence Ingestion Agent
            context_output: Output from Medical Context Agent
            causal_output: Output from Causal Inference Agent
            contradiction_output: Output from Contradiction Resolution Agent
            
        Returns:
            Constructed CausalGraph
        """
        logger.info("Building causal graph from agent results")
        
        graph = CausalGraph()
        
        # Step 1: Add symptom nodes from evidence
        symptoms = evidence_output.get("symptoms", [])
        symptom_nodes = {}
        for symptom in symptoms:
            node_id = graph.add_symptom(symptom, confidence=0.85)
            symptom_nodes[symptom] = node_id
        
        # Step 2: Add diagnosis nodes from evidence
        diagnoses = evidence_output.get("diagnoses", [])
        diagnosis_nodes = {}
        for diagnosis in diagnoses:
            node_id = graph.add_diagnosis(diagnosis, confidence=0.80)
            diagnosis_nodes[diagnosis] = node_id
        
        # Step 3: Add treatment nodes from evidence
        treatments = evidence_output.get("treatments", [])
        treatment_nodes = {}
        for treatment in treatments:
            node_id = graph.add_treatment(treatment, confidence=0.75)
            treatment_nodes[treatment] = node_id
        
        # Step 4: Add outcome nodes from evidence
        outcomes = evidence_output.get("outcomes", [])
        outcome_nodes = {}
        for outcome in outcomes:
            node_id = graph.add_outcome(outcome, confidence=0.70)
            outcome_nodes[outcome] = node_id
        
        # Step 5: Add evidence nodes from context
        evidence_nodes = {}
        for i, doc in enumerate(context_output[:3]):  # Top 3 documents
            content = doc.get("text", "")[:100]  # Truncate
            node_id = graph.add_evidence(
                content,
                confidence=doc.get("score", 0.5),
                metadata={"source": doc.get("metadata", {})}
            )
            evidence_nodes[i] = node_id
        
        # Step 6: Add causal edges from causal inference output
        causal_chains = causal_output.get("causal_chains", [])
        for chain in causal_chains:
            from_entity = chain.get("from", "")
            to_entity = chain.get("to", "")
            relationship = chain.get("relationship", "")
            confidence = chain.get("confidence", 0.5)
            
            # Find source and target nodes
            source_id = self._find_node_id(from_entity, symptom_nodes, diagnosis_nodes, treatment_nodes, outcome_nodes)
            target_id = self._find_node_id(to_entity, symptom_nodes, diagnosis_nodes, treatment_nodes, outcome_nodes)
            
            if source_id and target_id:
                # Determine edge type based on relationship
                edge_type = self._determine_edge_type(relationship)
                
                # Determine causal strength
                causal_strength = self._determine_causal_strength(confidence)
                
                try:
                    graph.add_edge(
                        source=source_id,
                        target=target_id,
                        edge_type=edge_type,
                        confidence=confidence,
                        causal_strength=causal_strength,
                        evidence_refs=[chain.get("evidence", "")],
                        reasoning_type="llm_based"
                    )
                except ValueError as e:
                    logger.warning(f"Failed to add edge: {e}")
        
        # Step 7: Connect evidence to diagnoses/treatments
        for i, evidence_id in evidence_nodes.items():
            # Connect to first diagnosis (simplified)
            if diagnosis_nodes:
                first_diagnosis = list(diagnosis_nodes.values())[0]
                graph.add_edge(
                    source=evidence_id,
                    target=first_diagnosis,
                    edge_type=EdgeType.SUPPORTS,
                    confidence=0.7,
                    causal_strength=CausalStrength.MODERATE
                )
        
        # Step 8: Validate and prune graph
        graph.prune_low_confidence(threshold=0.3)
        
        # Step 9: Log statistics
        stats = graph.get_stats()
        logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Step 10: Check for issues
        if graph.has_cycles():
            logger.warning("Graph contains cycles - may indicate circular reasoning")
        
        if not graph.is_connected():
            logger.warning("Graph is not connected - may have isolated components")
        
        return graph
    
    def _find_node_id(
        self,
        entity: str,
        *node_dicts: Dict[str, str]
    ) -> Optional[str]:
        """Find node ID for entity across multiple node dictionaries"""
        entity_lower = entity.lower()
        
        for node_dict in node_dicts:
            for key, node_id in node_dict.items():
                if key.lower() == entity_lower or entity_lower in key.lower():
                    return node_id
        
        return None
    
    def _determine_edge_type(self, relationship: str) -> EdgeType:
        """Determine edge type from relationship string"""
        relationship_lower = relationship.lower()
        
        if "cause" in relationship_lower or "leads to" in relationship_lower:
            return EdgeType.CAUSES
        elif "treat" in relationship_lower:
            return EdgeType.TREATED_BY
        elif "result" in relationship_lower or "outcome" in relationship_lower:
            return EdgeType.LEADS_TO
        else:
            return EdgeType.SUPPORTS
    
    def _determine_causal_strength(self, confidence: float) -> CausalStrength:
        """Determine causal strength from confidence score"""
        if confidence >= 0.8:
            return CausalStrength.STRONG
        elif confidence >= 0.6:
            return CausalStrength.MODERATE
        else:
            return CausalStrength.WEAK


# Singleton instance
_builder_instance: Optional[GraphBuilder] = None


def get_graph_builder() -> GraphBuilder:
    """Get or create singleton GraphBuilder instance"""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = GraphBuilder()
    return _builder_instance
