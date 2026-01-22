"""
Specialized Agents - Medical Domain Focus
Evidence Ingestion, Medical Context, Causal Inference, Contradiction Resolution
"""

from typing import Dict, Any, List, Optional
import re
import json
from agents.base_agent import BaseAgent
from core.llm_manager import get_llm_manager
from storage.vector_store import get_vector_store
from core.logging_config import get_logger

logger = get_logger("specialized_agents")


class EvidenceIngestionAgent(BaseAgent):
    """
    Extracts structured medical data from raw inputs.
    Focuses on symptoms, diagnoses, treatments, outcomes.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="evidence_ingestion",
            description="Extracts structured medical data from inputs"
        )
        self.llm = get_llm_manager()
    
    def ingest(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input has required fields"""
        if "text" not in input_data:
            raise ValueError("Input must contain 'text' field")
        
        return {
            "raw_text": input_data["text"],
            "source": input_data.get("source", "unknown"),
            "metadata": input_data.get("metadata", {})
        }
    
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured medical information using LLM"""
        raw_text = processed_input["raw_text"]
        
        system_message = """You are a medical information extraction expert. 
Extract structured medical information from the provided text.
Identify: symptoms, diagnoses, treatments, medications, test results, and outcomes.
Return a JSON object with these categories."""
        
        prompt = f"""Extract medical information from this text:

{raw_text}

Return a JSON object with the following structure:
{{
    "symptoms": ["list of symptoms"],
    "diagnoses": ["list of diagnoses"],
    "treatments": ["list of treatments"],
    "medications": ["list of medications"],
    "test_results": ["list of test results"],
    "outcomes": ["list of outcomes"],
    "temporal_info": "any temporal information (dates, durations)"
}}"""
        
       # Call LLM with JSON response format
        response = self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,  # Lower temperature for extraction
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        try:
            extracted_data = json.loads(response["output"])
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON, using fallback")
            extracted_data = {
                "symptoms": [],
                "diagnoses": [],
                "treatments": [],
                "medications": [],
                "test_results": [],
                "outcomes": []
            }
        
        return {
            "extracted_data": extracted_data,
            "raw_text": raw_text,
            "confidence": 0.8,  # TODO: Implement confidence scoring
            "conclusions": [f"Extracted {sum(len(v) for v in extracted_data.values() if isinstance(v, list))} medical entities"]
        }
    
    def output(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format extracted data"""
        return reasoning_result["extracted_data"]


class MedicalContextAgent(BaseAgent):
    """
    Retrieves relevant medical knowledge from vector store (RAG).
    Provides medical guidelines, research, and contextual information.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="medical_context",
            description="Retrieves medical knowledge via RAG"
        )
        self.vector_store = get_vector_store()
    
    def ingest(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input contains query"""
        if "query" not in input_data and "extracted_data" not in input_data:
            raise ValueError("Input must contain 'query' or 'extracted_data'")
        
        return {
            "query": input_data.get("query", ""),
            "extracted_data": input_data.get("extracted_data", {}),
            "top_k": input_data.get("top_k", 5)
        }
    
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant medical context"""
        # Build query from extracted data or use direct query
        if processed_input["extracted_data"]:
            query_parts = []
            extracted = processed_input["extracted_data"]
            
            for category in ["symptoms", "diagnoses", "treatments"]:
                if category in extracted and extracted[category]:
                    query_parts.extend(extracted[category])
            
            query = " ".join(query_parts)
        else:
            query = processed_input["query"]
        
        if not query:
            return {
                "context_documents": [],
                "confidence": 0.0,
                "conclusions": ["No query provided"]
            }
        
        # Retrieve from vector store
        self.logger.info(f"Retrieving context for: {query[:100]}...")
        results = self.vector_store.retrieve(
            query=query,
            top_k=processed_input["top_k"],
            namespace="medical_knowledge"
        )
        
        return {
            "context_documents": results,
            "query": query,
            "confidence": 0.9 if len(results) > 0 else 0.3,
            "conclusions": [f"Retrieved {len(results)} relevant documents"]
        }
    
    def output(self, reasoning_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format context documents"""
        return reasoning_result["context_documents"]


class CausalInferenceAgent(BaseAgent):
    """
    Builds cause-effect relationships for medical reasoning.
    Maps: symptoms -> diagnosis -> treatment -> outcome chains.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="causal_inference",
            description="Builds medical cause-effect relationships"
        )
        self.llm = get_llm_manager()
    
    def ingest(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input contains medical data"""
        if "extracted_data" not in input_data:
            raise ValueError("Input must contain 'extracted_data'")
        
        return {
            "extracted_data": input_data["extracted_data"],
            "context_documents": input_data.get("context_documents", [])
        }
    
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Build causal chains using LLM"""
        extracted = processed_input["extracted_data"]
        context_docs = processed_input["context_documents"]
        
        # Format context
        context_text = "\n\n".join([
            f"Document {i+1}: {doc.get('text', '')[:200]}..."
            for i, doc in enumerate(context_docs[:3])
        ])
        
        system_message = """You are a medical reasoning expert specializing in causal inference.
Build cause-effect chains showing how symptoms lead to diagnoses, and how treatments affect outcomes.
Provide confidence scores (0-1) for each causal link."""
        
        prompt = f"""Given this medical information:

Symptoms: {', '.join(extracted.get('symptoms', []))}
Diagnoses: {', '.join(extracted.get('diagnoses', []))}
Treatments: {', '.join(extracted.get('treatments', []))}
Outcomes: {', '.join(extracted.get('outcomes', []))}

And this contextual medical knowledge:
{context_text if context_text else 'No additional context provided'}

Build causal chains showing relationships. Return JSON:
{{
    "causal_chains": [
        {{
            "from": "cause",
            "to": "effect",
            "relationship": "leads to|caused by|treated by",
            "confidence": 0.0-1.0,
            "evidence": "supporting evidence"
        }}
    ],
    "overall_confidence": 0.0-1.0,
    "uncertainties": ["list any uncertainties"]
}}"""
        
        response = self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.4,
            response_format={"type": "json_object"}
        )
        
        try:
            causal_data = json.loads(response["output"])
        except json.JSONDecodeError:
            causal_data = {
                "causal_chains": [],
                "overall_confidence": 0.5,
                "uncertainties": ["Failed to parse LLM response"]
            }
        
        return {
            "causal_chains": causal_data.get("causal_chains", []),
            "uncertainties": causal_data.get("uncertainties", []),
            "confidence": causal_data.get("overall_confidence", 0.5),
            "conclusions": [f"Identified {len(causal_data.get('causal_chains', []))} causal relationships"]
        }
    
    def output(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format causal chains"""
        return {
            "causal_chains": reasoning_result["causal_chains"],
            "uncertainties": reasoning_result["uncertainties"],
            "confidence": reasoning_result["confidence"]
        }


class ContradictionResolutionAgent(BaseAgent):
    """
    Identifies and resolves conflicting medical evidence.
    Handles contradictions in diagnoses, treatments, or outcomes.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="contradiction_resolution",
            description="Resolves conflicting medical evidence"
        )
        self.llm = get_llm_manager()
    
    def ingest(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input contains causal chains or evidence"""
        if "causal_chains" not in input_data and "evidence_list" not in input_data:
            raise ValueError("Input must contain 'causal_chains' or 'evidence_list'")
        
        return {
            "causal_chains": input_data.get("causal_chains", []),
            "evidence_list": input_data.get("evidence_list", []),
            "context_documents": input_data.get("context_documents", [])
        }
    
    def reason(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and resolve contradictions"""
        causal_chains = processed_input["causal_chains"]
        
        if not causal_chains:
            return {
                "contradictions": [],
                "resolutions": [],
                "confidence": 1.0,
                "conclusions": ["No contradictions found"]
            }
        
        # Format causal chains for analysis
        chains_text = "\n".join([
            f"{i+1}. {chain.get('from', '')} → {chain.get('to', '')} "
            f"(confidence: {chain.get('confidence', 0):.2f})"
            for i, chain in enumerate(causal_chains)
        ])
        
        system_message = """You are a medical expert specializing in evidence analysis.
Identify contradictions or conflicts in medical reasoning.
Resolve conflicts by weighing evidence quality and confidence scores."""
        
        prompt = f"""Analyze these causal relationships for contradictions:

{chains_text}

Identify any contradictions (e.g., conflicting diagnoses, incompatible treatments).
For each contradiction, propose a resolution. Return JSON:
{{
    "contradictions": [
        {{
            "statement_1": "first conflicting statement",
            "statement_2": "second conflicting statement",
            "type": "diagnosis|treatment|outcome",
            "severity": "high|medium|low"
        }}
    ],
    "resolutions": [
        {{
            "contradiction_index": 0,
            "resolution": "proposed resolution",
            "rationale": "reasoning",
            "confidence": 0.0-1.0
        }}
    ],
    "overall_confidence": 0.0-1.0
}}"""
        
        response = self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        try:
            contradiction_data = json.loads(response["output"])
        except json.JSONDecodeError:
            contradiction_data = {
                "contradictions": [],
                "resolutions": [],
                "overall_confidence": 0.5
            }
        
        return {
            "contradictions": contradiction_data.get("contradictions", []),
            "resolutions": contradiction_data.get("resolutions", []),
            "confidence": contradiction_data.get("overall_confidence", 0.5),
            "conclusions": [
                f"Found {len(contradiction_data.get('contradictions', []))} contradictions, "
                f"resolved {len(contradiction_data.get('resolutions', []))}"
            ]
        }
    
    def output(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format contradiction analysis"""
        return {
            "contradictions": reasoning_result["contradictions"],
            "resolutions": reasoning_result["resolutions"],
            "confidence": reasoning_result["confidence"]
        }
