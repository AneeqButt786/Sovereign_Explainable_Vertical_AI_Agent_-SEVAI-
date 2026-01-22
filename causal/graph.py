"""
Causal Graph - Medical Domain Reasoning Graph Structure
NetworkX-based graph for representing causal relationships in medical reasoning
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import networkx as nx
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import hashlib
from core.logging_config import get_logger

logger = get_logger("causal_graph")


class NodeType(Enum):
    """Types of nodes in medical causal graph"""
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    EVIDENCE = "evidence"


class EdgeType(Enum):
    """Types of edges in medical causal graph"""
    CAUSES = "causes"  # Symptom → Diagnosis
    TREATED_BY = "treated_by"  # Diagnosis → Treatment
    LEADS_TO = "leads_to"  # Treatment → Outcome
    SUPPORTS = "supports"  # Evidence → Any


class CausalStrength(Enum):
    """Strength of causal relationship"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class CausalNode:
    """Node in the causal graph"""
    node_id: str
    node_type: NodeType
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CausalEdge:
    """Edge in the causal graph"""
    source: str
    target: str
    edge_type: EdgeType
    confidence: float
    causal_strength: CausalStrength
    evidence_refs: List[str] = field(default_factory=list)
    reasoning_type: str = "symbolic"  # symbolic, probabilistic, llm_based
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary"""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "causal_strength": self.causal_strength.value,
            "evidence_refs": self.evidence_refs,
            "reasoning_type": self.reasoning_type,
            "metadata": self.metadata
        }


class CausalGraph:
    """
    Medical domain causal graph using NetworkX.
    Represents cause-effect relationships in medical reasoning.
    """
    
    def __init__(self, graph_id: Optional[str] = None):
        """
        Initialize causal graph
        
        Args:
            graph_id: Unique identifier for this graph
        """
        self.graph_id = graph_id or self._generate_id()
        self.graph = nx.DiGraph()
        self.created_at = datetime.now(timezone.utc)
        self.metadata = {}
        
        logger.info(f"Causal graph initialized: {self.graph_id}")
    
    def _generate_id(self) -> str:
        """Generate unique graph ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _generate_node_id(self, content: str, node_type: NodeType) -> str:
        """Generate unique node ID"""
        content_hash = hashlib.md5(f"{node_type.value}:{content}".encode()).hexdigest()[:8]
        return f"{node_type.value}_{content_hash}"
    
    def add_node(
        self,
        content: str,
        node_type: NodeType,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ) -> str:
        """
        Add a node to the graph
        
        Args:
            content: Content of the node (e.g., "fever", "pneumonia")
            node_type: Type of node
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            node_id: Optional custom node ID
            
        Returns:
            Node ID
        """
        if node_id is None:
            node_id = self._generate_node_id(content, node_type)
        
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.graph.add_node(node_id, **node.to_dict())
        logger.debug(f"Added node: {node_id} ({node_type.value})")
        
        return node_id
    
    def add_symptom(self, content: str, confidence: float, **kwargs) -> str:
        """Add a symptom node"""
        return self.add_node(content, NodeType.SYMPTOM, confidence, **kwargs)
    
    def add_diagnosis(self, content: str, confidence: float, **kwargs) -> str:
        """Add a diagnosis node"""
        return self.add_node(content, NodeType.DIAGNOSIS, confidence, **kwargs)
    
    def add_treatment(self, content: str, confidence: float, **kwargs) -> str:
        """Add a treatment node"""
        return self.add_node(content, NodeType.TREATMENT, confidence, **kwargs)
    
    def add_outcome(self, content: str, confidence: float, **kwargs) -> str:
        """Add an outcome node"""
        return self.add_node(content, NodeType.OUTCOME, confidence, **kwargs)
    
    def add_evidence(self, content: str, confidence: float, **kwargs) -> str:
        """Add an evidence node"""
        return self.add_node(content, NodeType.EVIDENCE, confidence, **kwargs)
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        confidence: float,
        causal_strength: CausalStrength = CausalStrength.MODERATE,
        evidence_refs: Optional[List[str]] = None,
        reasoning_type: str = "symbolic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Add an edge to the graph
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge
            confidence: Confidence score (0-1)
            causal_strength: Strength of causal relationship
            evidence_refs: References to supporting evidence
            reasoning_type: Type of reasoning used
            metadata: Additional metadata
            
        Returns:
            Tuple of (source, target)
        """
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        if target not in self.graph.nodes:
            raise ValueError(f"Target node {target} not in graph")
        
        edge = CausalEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            confidence=confidence,
            causal_strength=causal_strength,
            evidence_refs=evidence_refs or [],
            reasoning_type=reasoning_type,
            metadata=metadata or {}
        )
        
        self.graph.add_edge(source, target, **edge.to_dict())
        logger.debug(f"Added edge: {source} → {target} ({edge_type.value})")
        
        return (source, target)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data"""
        if node_id not in self.graph.nodes:
            return None
        return dict(self.graph.nodes[node_id])
    
    def get_edge(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Get edge data"""
        if not self.graph.has_edge(source, target):
            return None
        return dict(self.graph.edges[source, target])
    
    def get_all_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Find all paths from source to target
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path from source to target"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
    
    def find_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Find all nodes of a specific type"""
        return [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") == node_type.value
        ]
    
    def has_cycles(self) -> bool:
        """Check if graph has cycles (should not in valid causal graph)"""
        try:
            nx.find_cycle(self.graph)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def is_connected(self) -> bool:
        """Check if graph is weakly connected"""
        return nx.is_weakly_connected(self.graph)
    
    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """
        Remove edges with confidence below threshold
        
        Args:
            threshold: Minimum confidence to keep edge
            
        Returns:
            Number of edges removed
        """
        edges_to_remove = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("confidence", 1.0) < threshold
        ]
        
        self.graph.remove_edges_from(edges_to_remove)
        logger.info(f"Pruned {len(edges_to_remove)} low-confidence edges")
        
        return len(edges_to_remove)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        return {
            "graph_id": self.graph_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "nodes": [
                {"id": node_id, **data}
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export graph to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_graphml(self, filepath: str):
        """Export graph to GraphML format (for Gephi, Cytoscape)"""
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Graph exported to GraphML: {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        """Load graph from dictionary"""
        graph = cls(graph_id=data["graph_id"])
        graph.created_at = datetime.fromisoformat(data["created_at"])
        graph.metadata = data["metadata"]
        
        # Add nodes
        for node_data in data["nodes"]:
            node_id = node_data.pop("id")
            graph.graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in data["edges"]:
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.graph.add_edge(source, target, **edge_data)
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> "CausalGraph":
        """Load graph from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "graph_id": self.graph_id,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "nodes_by_type": {
                nt.value: len(self.find_nodes_by_type(nt))
                for nt in NodeType
            },
            "has_cycles": self.has_cycles(),
            "is_connected": self.is_connected(),
            "density": nx.density(self.graph)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<CausalGraph(id='{self.graph_id}', nodes={stats['num_nodes']}, edges={stats['num_edges']})>"
