"""
Trail Extractor - Causal Reasoning Trail Extraction and Visualization
Extract reasoning paths and generate explanations
"""

from typing import Dict, Any, List, Optional, Tuple
from causal.graph import CausalGraph, NodeType
from causal.confidence import get_confidence_scorer
from core.logging_config import get_logger

logger = get_logger("trail_extractor")


class TrailExtractor:
    """
    Extracts reasoning trails from causal graphs.
    Generates human-readable explanations and visualization data.
    """
    
    def __init__(self):
        """Initialize trail extractor"""
        self.confidence_scorer = get_confidence_scorer()
        logger.info("Trail extractor initialized")
    
    def extract(self, graph: CausalGraph) -> Dict[str, Any]:
        """
        Extract complete reasoning trail from graph
        
        Args:
            graph: Causal graph to extract from
            
        Returns:
            Dictionary containing reasoning trail data
        """
        logger.info(f"Extracting reasoning trail from graph {graph.graph_id}")
        
        # Find symptom and outcome nodes
        symptom_nodes = graph.find_nodes_by_type(NodeType.SYMPTOM)
        outcome_nodes = graph.find_nodes_by_type(NodeType.OUTCOME)
        
        # Extract all paths from symptoms to outcomes
        all_paths = []
        for symptom in symptom_nodes:
            for outcome in outcome_nodes:
                paths = graph.get_all_paths(symptom, outcome)
                for path in paths:
                    all_paths.append({
                        "path": path,
                        "start": symptom,
                        "end": outcome,
                        "length": len(path)
                    })
        
        # Generate narrative for each path
        narratives = []
        for path_data in all_paths[:5]:  # Limit to top 5 paths
            narrative = self._generate_narrative(graph, path_data["path"])
            narratives.append(narrative)
        
        # Extract key steps
        key_steps = self._extract_key_steps(graph, all_paths)
        
        trail = {
            "graph_id": graph.graph_id,
            "num_paths": len(all_paths),
            "paths": all_paths[:10],  # Include top 10 paths
            "narratives": narratives,
            "key_steps": key_steps,
            "graph_stats": graph.get_stats()
        }
        
        logger.info(f"Extracted {len(all_paths)} reasoning paths")
        
        return trail
    
    def _generate_narrative(self, graph: CausalGraph, path: List[str]) -> str:
        """
        Generate human-readable narrative for a reasoning path
        
        Args:
            graph: Causal graph
            path: List of node IDs in path
            
        Returns:
            Narrative string
        """
        if not path:
            return "No reasoning path available."
        
        narrative_parts = []
        
        for i, node_id in enumerate(path):
            node_data = graph.get_node(node_id)
            if not node_data:
                continue
            
            content = node_data.get("content", "Unknown")
            confidence = node_data.get("confidence", 0.0)
            node_type = node_data.get("node_type", "unknown")
            
            # Format based on node type
            if node_type == NodeType.SYMPTOM.value:
                narrative_parts.append(f"Patient presented with {content} (confidence: {confidence:.0%})")
            elif node_type == NodeType.DIAGNOSIS.value:
                narrative_parts.append(f"Diagnosed with {content} (confidence: {confidence:.0%})")
            elif node_type == NodeType.TREATMENT.value:
                narrative_parts.append(f"Treatment: {content} (confidence: {confidence:.0%})")
            elif node_type == NodeType.OUTCOME.value:
                narrative_parts.append(f"Expected outcome: {content} (confidence: {confidence:.0%})")
            elif node_type == NodeType.EVIDENCE.value:
                narrative_parts.append(f"Supporting evidence: {content[:50]}...")
            
            # Add edge information if not last node
            if i < len(path) - 1:
                next_node_id = path[i + 1]
                edge_data = graph.get_edge(node_id, next_node_id)
                if edge_data:
                    edge_type = edge_data.get("edge_type", "related to")
                    edge_confidence = edge_data.get("confidence", 0.0)
                    narrative_parts.append(f"  └─ ({edge_type}, confidence: {edge_confidence:.0%})")
        
        return "\n".join(narrative_parts)
    
    def _extract_key_steps(
        self,
        graph: CausalGraph,
        all_paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key reasoning steps from paths"""
        # Find most common nodes across all paths
        node_frequency = {}
        
        for path_data in all_paths:
            for node_id in path_data["path"]:
                node_frequency[node_id] = node_frequency.get(node_id, 0) + 1
        
        # Sort by frequency
        sorted_nodes = sorted(
            node_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extract top nodes askey steps
        key_steps = []
        for node_id, frequency in sorted_nodes[:5]:
            node_data = graph.get_node(node_id)
            if node_data:
                key_steps.append({
                    "node_id": node_id,
                    "content": node_data.get("content"),
                    "type": node_data.get("node_type"),
                    "confidence": node_data.get("confidence"),
                    "frequency": frequency
                })
        
        return key_steps
    
    def export_graph_json(self, graph: CausalGraph) -> str:
        """
        Export graph as JSON for visualization (D3.js, React Flow)
        
        Args:
            graph: Causal graph to export
            
        Returns:
            JSON string
        """
        return graph.to_json()
    
    def export_graph_for_react_flow(self, graph: CausalGraph) -> Dict[str, Any]:
        """
        Export graph in React Flow format
        
        Args:
            graph: Causal graph
            
        Returns:
            Dict with 'nodes' and 'edges' for React Flow
        """
        react_flow_nodes = []
        react_flow_edges = []
        
        # Convert nodes
        for node_id, node_data in graph.graph.nodes(data=True):
            react_flow_nodes.append({
                "id": node_id,
                "data": {
                    "label": node_data.get("content", ""),
                    "confidence": node_data.get("confidence", 0.0),
                    "type": node_data.get("node_type", "")
                },
                "type": node_data.get("node_type", "default"),
                "position": {"x": 0, "y": 0}  # Layout will be computed client-side
            })
        
        # Convert edges
        for source, target, edge_data in graph.graph.edges(data=True):
            react_flow_edges.append({
                "id": f"{source}-{target}",
                "source": source,
                "target": target,
                "label": edge_data.get("edge_type", ""),
                "data": {
                    "confidence": edge_data.get("confidence", 0.0),
                    "strength": edge_data.get("causal_strength", "")
                },
                "animated": edge_data.get("confidence", 0.0) > 0.8
            })
        
        return {
            "nodes": react_flow_nodes,
            "edges": react_flow_edges
        }
    
    def generate_summary(self, graph: CausalGraph, trail: Dict[str, Any]) -> str:
        """
        Generate executive summary of reasoning trail
        
        Args:
            graph: Causal graph
            trail: Trail extraction result
            
        Returns:
            Summary string
        """
        stats = trail.get("graph_stats", {})
        key_steps = trail.get("key_steps", [])
        
        summary = f"Reasoning Summary for {graph.graph_id}\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Graph Statistics:\n"
        summary += f"  - Total nodes: {stats.get('num_nodes', 0)}\n"
        summary += f"  - Total edges: {stats.get('num_edges', 0)}\n"
        summary += f"  - Reasoning paths: {trail.get('num_paths', 0)}\n"
        summary += f"  - Connected: {'Yes' if stats.get('is_connected') else 'No'}\n"
        summary += f"  - Has cycles: {'Yes' if stats.get('has_cycles') else 'No'}\n\n"
        
        summary += "Key Reasoning Steps:\n"
        for i, step in enumerate(key_steps, 1):
            summary += f"  {i}. {step.get('content')} "
            summary += f"({step.get('type')}, confidence: {step.get('confidence', 0):.0%})\n"
        
        summary += "\n" + "=" * 60 + "\n"
        
        return summary


# Singleton instance
_extractor_instance: Optional[TrailExtractor] = None


def get_trail_extractor() -> TrailExtractor:
    """Get or create singleton TrailExtractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = TrailExtractor()
    return _extractor_instance
