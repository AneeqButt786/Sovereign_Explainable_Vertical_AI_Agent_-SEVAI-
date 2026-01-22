"""
Bias Detector - Fairness and Bias Detection in Medical Reasoning
Detects potential demographic biases and generates counterfactuals
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from core.logging_config import get_logger
from causal.graph import CausalGraph, NodeType

logger = get_logger("bias_detector")


@dataclass
class BiasReport:
    """Report on detected biases"""
    has_bias: bool
    bias_score: float  # 0-1 (higher is more biased)
    detected_types: List[str]
    details: Dict[str, Any]
    counterfactuals: List[Dict[str, Any]]
    recommendations: List[str]


class BiasDetector:
    """
    Detects potential biases in medical reasoning execution.
    Focuses on demographic fairness and anchoring bias.
    """
    
    # Sensitive attributes to check
    SENSITIVE_ATTRIBUTES = ["age", "gender", "sex", "race", "ethnicity", "socioeconomic_status"]
    
    def __init__(self):
        """Initialize bias detector"""
        logger.info("Bias detector initialized")
    
    def check_graph(self, graph: CausalGraph, input_metadata: Dict[str, Any]) -> BiasReport:
        """
        Check causal graph for potential biases
        
        Args:
            graph: Constructed causal graph
            input_metadata: Metadata about the patient/case
            
        Returns:
            BiasReport
        """
        logger.info(f"Checking graph {graph.graph_id} for bias")
        
        detected_types = []
        details = {}
        recommendations = []
        
        # 1. Check for demographic usage in reasoning
        demographic_usage = self._check_demographic_usage(graph, input_metadata)
        if demographic_usage["used_explicitly"]:
            # It's not always bias to use demographics (e.g. breast cancer in women), 
            # but it should be flagged for review if high impact
            details["demographic_usage"] = demographic_usage
            if demographic_usage["impact_score"] > 0.7:
                detected_types.append("high_demographic_impact")
                recommendations.append("Verify demographic factors are clinically relevant")
        
        # 2. Check for premature closure / anchoring
        # If graph is very sparse or linear with high confidence early
        if self._check_premature_closure(graph):
            detected_types.append("premature_closure")
            details["premature_closure"] = True
            recommendations.append("Consider alternative diagnoses (premature closure detected)")
        
        # 3. Generate counterfactual suggestions
        counterfactuals = self._generate_counterfactual_suggestions(input_metadata)
        
        # Calculate overall score
        input_has_demographics = any(k in input_metadata for k in self.SENSITIVE_ATTRIBUTES)
        has_bias = len(detected_types) > 0
        bias_score = 0.0
        
        if has_bias:
            bias_score = 0.5 + (0.1 * len(detected_types))
        
        return BiasReport(
            has_bias=has_bias,
            bias_score=min(1.0, bias_score),
            detected_types=detected_types,
            details=details,
            counterfactuals=counterfactuals,
            recommendations=recommendations
        )
    
    def _check_demographic_usage(
        self, 
        graph: CausalGraph, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if/how demographics were used in the graph nodes/edges"""
        used_explicitly = False
        impact_score = 0.0
        used_attributes = []
        
        # Scan nodes for sensitive keywords
        for node_id, node_data in graph.graph.nodes(data=True):
            content = node_data.get("content", "").lower()
            
            for attr in self.SENSITIVE_ATTRIBUTES:
                # check if attribute value is in content
                attr_val = str(metadata.get(attr, "")).lower()
                if attr_val and attr_val in content:
                    used_explicitly = True
                    used_attributes.append(attr)
                    # Higher confidence nodes using demographics = higher impact
                    impact_score = max(impact_score, node_data.get("confidence", 0.0))
        
        return {
            "used_explicitly": used_explicitly,
            "used_attributes": list(set(used_attributes)),
            "impact_score": impact_score
        }
    
    def _check_premature_closure(self, graph: CausalGraph) -> bool:
        """
        Check for premature closure (anchoring).
        Heuristic: High confidence diagnosis reached with very few evidence steps.
        """
        # Count evidence nodes
        evidence_nodes = graph.find_nodes_by_type(NodeType.EVIDENCE)
        diagnosis_nodes = graph.find_nodes_by_type(NodeType.DIAGNOSIS)
        
        if not diagnosis_nodes:
            return False
            
        # Get highest confidence diagnosis
        max_diag_conf = 0.0
        for nid in diagnosis_nodes:
            node = graph.get_node(nid)
            if node:
                max_diag_conf = max(max_diag_conf, node.get("confidence", 0.0))
        
        # Heuristic: High confidence (>0.8) but low evidence count (<2)
        if max_diag_conf > 0.8 and len(evidence_nodes) < 2:
            return True
            
        return False
    
    def _generate_counterfactual_suggestions(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions for counterfactual analysis.
        e.g. "Run check with gender=Female" if current is Male
        """
        suggestions = []
        
        if "gender" in metadata:
            curr = str(metadata["gender"]).lower()
            if curr in ["male", "m"]:
                suggestions.append({"attribute": "gender", "original": curr, "counterfactual": "female"})
            elif curr in ["female", "f"]:
                suggestions.append({"attribute": "gender", "original": curr, "counterfactual": "male"})
                
        if "age" in metadata:
            curr_age = int(metadata["age"])
            if curr_age < 18:
                suggestions.append({"attribute": "age", "original": curr_age, "counterfactual": 35})
            elif curr_age > 65:
                 suggestions.append({"attribute": "age", "original": curr_age, "counterfactual": 45})
        
        return suggestions


# Singleton instance
_detector_instance: Optional[BiasDetector] = None


def get_bias_detector() -> BiasDetector:
    """Get or create singleton BiasDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BiasDetector()
    return _detector_instance
