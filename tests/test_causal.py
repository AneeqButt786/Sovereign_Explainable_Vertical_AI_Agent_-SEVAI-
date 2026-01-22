"""
Tests for Phase 3 causal inference components
"""

import pytest
from causal.graph import CausalGraph, NodeType, EdgeType, CausalStrength
from causal.confidence import get_confidence_scorer, ConfidenceFactors
from causal.graph_builder import get_graph_builder
from causal.trail_extractor import get_trail_extractor


class TestCausalGraph:
    """Test causal graph structure"""
    
    def test_node_creation(self):
        """Test creating nodes in graph"""
        graph = CausalGraph()
        
        # Add symptom node
        node_id = graph.add_symptom("fever", confidence=0.9)
        assert node_id is not None
        assert "symptom" in node_id
        
        # Verify node data
        node_data = graph.get_node(node_id)
        assert node_data["content"] == "fever"
        assert node_data["confidence"] == 0.9
        assert node_data["node_type"] == NodeType.SYMPTOM.value
    
    def test_edge_creation(self):
        """Test creating edges between nodes"""
        graph = CausalGraph()
        
        # Add nodes
        symptom_id = graph.add_symptom("fever", confidence=0.9)
        diagnosis_id = graph.add_diagnosis("pneumonia", confidence=0.85)
        
        # Add edge
        source, target = graph.add_edge(
            source=symptom_id,
            target=diagnosis_id,
            edge_type=EdgeType.CAUSES,
            confidence=0.8,
            causal_strength=CausalStrength.STRONG
        )
        
        assert source == symptom_id
        assert target == diagnosis_id
        
        # Verify edge data
        edge_data = graph.get_edge(symptom_id, diagnosis_id)
        assert edge_data["confidence"] == 0.8
        assert edge_data["edge_type"] == EdgeType.CAUSES.value
    
    def test_path_finding(self):
        """Test finding paths in graph"""
        graph = CausalGraph()
        
        # Build simple chain: symptom → diagnosis → treatment
        s = graph.add_symptom("fever", 0.9)
        d = graph.add_diagnosis("pneumonia", 0.85)
        t = graph.add_treatment("antibiotics", 0.8)
        
        graph.add_edge(s, d, EdgeType.CAUSES, 0.85)
        graph.add_edge(d, t, EdgeType.TREATED_BY, 0.9)
        
        # Find path
        paths = graph.get_all_paths(s, t)
        assert len(paths) > 0
        assert paths[0] == [s, d, t]
    
    def test_prune_low_confidence(self):
        """Test pruning low-confidence edges"""
        graph = CausalGraph()
        
        s = graph.add_symptom("fever", 0.9)
        d = graph.add_diagnosis("pneumonia", 0.85)
        
        # Add low-confidence edge
        graph.add_edge(s, d, EdgeType.CAUSES, 0.2)  # Below threshold
        
        # Prune
        removed = graph.prune_low_confidence(threshold=0.3)
        assert removed == 1
        assert not graph.graph.has_edge(s, d)
    
    def test_json_export(self):
        """Test JSON export and import"""
        graph = CausalGraph()
        s = graph.add_symptom("fever", 0.9)
        
        # Export to JSON
        json_str = graph.to_json()
        assert "fever" in json_str
        
        # Import from JSON
        graph2 = CausalGraph.from_json(json_str)
        assert graph2.graph_id == graph.graph_id
        assert len(list(graph2.graph.nodes())) == len(list(graph.graph.nodes()))


class TestConfidenceScorer:
    """Test confidence scoring system"""
    
    def test_single_step_confidence(self):
        """Test confidence calculation for single step"""
        scorer = get_confidence_scorer()
        
        confidence = scorer.calculate(
            evidence_quality=0.8,
            reasoning_coherence=0.9,
            llm_confidence=0.7,
            context_match=0.85
        )
        
        assert 0.0 <= confidence <= 1.0
        # Should be weighted average
        expected = (0.8 * 0.4) + (0.9 * 0.3) + (0.7 * 0.2) + (0.85 * 0.1)
        assert abs(confidence - expected) < 0.01
    
    def test_chain_confidence_decay(self):
        """Test confidence decay for chains"""
        scorer = get_confidence_scorer()
        
        # Chain of 3 high-confidence steps
        chain = [0.9, 0.9, 0.9]
        
        # Without decay
        no_decay = scorer.aggregate_chain(chain, use_decay=False)
        assert no_decay == 0.9  # Minimum
        
        # With decay
        with_decay = scorer.aggregate_chain(chain, use_decay=True)
        assert with_decay < 0.9  # Should be lower due to decay
    
    def test_confidence_levels(self):
        """Test confidence level categorization"""
        scorer = get_confidence_scorer()
        
        assert scorer.get_level(0.85) == "high"
        assert scorer.get_level(0.70) == "medium"
        assert scorer.get_level(0.50) == "low"
        assert scorer.get_level(0.30) == "insufficient"
    
    def test_evidence_quality_calculation(self):
        """Test evidence quality factor calculation"""
        scorer = get_confidence_scorer()
        
        quality = scorer.calculate_evidence_quality(
            source_credibility=1.0,  # Peer-reviewed
            recency_score=0.9,  # Recent
            sample_size_score=0.8  # Large RCT
        )
        
        assert 0.8 <= quality <= 1.0


class TestGraphBuilder:
    """Test graph builder"""
    
    def test_build_from_results(self):
        """Test building graph from agent outputs"""
        builder = get_graph_builder()
        
        # Mock agent outputs
        evidence_output = {
            "symptoms": ["fever", "cough"],
            "diagnoses": ["pneumonia"],
            "treatments": ["antibiotics"],
            "outcomes": ["recovery"]
        }
        
        context_output = [
            {"text": "Pneumonia treatment guidelines...", "score": 0.8}
        ]
        
        causal_output = {
            "causal_chains": [
                {
                    "from": "fever",
                    "to": "pneumonia",
                    "relationship": "causes",
                    "confidence": 0.85,
                    "evidence": "Clinical observation"
                }
            ]
        }
        
        contradiction_output = {
            "contradictions": [],
            "resolutions": []
        }
        
        # Build graph
        graph = builder.build_from_results(
            evidence_output,
            context_output,
            causal_output,
            contradiction_output
        )
        
        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() >= 0


class TestTrailExtractor:
    """Test trail extractor"""
    
    def test_extract_trail(self):
        """Test extracting reasoning trail"""
        # Build simple graph
        graph = CausalGraph()
        s = graph.add_symptom("fever", 0.9)
        d = graph.add_diagnosis("pneumonia", 0.85)
        t = graph.add_treatment("antibiotics", 0.8)
        o = graph.add_outcome("recovery", 0.75)
        
        graph.add_edge(s, d, EdgeType.CAUSES, 0.85)
        graph.add_edge(d, t, EdgeType.TREATED_BY, 0.9)
        graph.add_edge(t, o, EdgeType.LEADS_TO, 0.8)
        
        # Extract trail
        extractor = get_trail_extractor()
        trail = extractor.extract(graph)
        
        assert "graph_id" in trail
        assert trail["num_paths"] > 0
        assert len(trail["narratives"]) > 0
    
    def test_react_flow_export(self):
        """Test React Flow format export"""
        graph = CausalGraph()
        s = graph.add_symptom("fever", 0.9)
        d = graph.add_diagnosis("pneumonia", 0.85)
        graph.add_edge(s, d, EdgeType.CAUSES, 0.85)
        
        extractor = get_trail_extractor()
        react_flow_data = extractor.export_graph_for_react_flow(graph)
        
        assert "nodes" in react_flow_data
        assert "edges" in react_flow_data
        assert len(react_flow_data["nodes"]) == 2
        assert len(react_flow_data["edges"]) == 1
