"""
Confidence Scoring System
Multi-factor confidence calculation for medical reasoning
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import math
from core.logging_config import get_logger

logger = get_logger("confidence")


class ConfidenceLevel(str):
    """Confidence level thresholds"""
    HIGH = "high"  # >= 0.80
    MEDIUM = "medium"  # 0.60-0.79
    LOW = "low"  # 0.40-0.59
    INSUFFICIENT = "insufficient"  # < 0.40


@dataclass
class ConfidenceFactors:
    """Individual confidence factors"""
    evidence_quality: float  # 0-1
    reasoning_coherence: float  # 0-1
    llm_confidence: float  # 0-1
    context_match: float  # 0-1


class ConfidenceScorer:
    """
    Multi-factor confidence scoring for medical reasoning.
    Combines evidence quality, reasoning coherence, LLM confidence, and context match.
    """
    
    # Factor weights (must sum to 1.0)
    WEIGHTS = {
        "evidence_quality": 0.40,
        "reasoning_coherence": 0.30,
        "llm_confidence": 0.20,
        "context_match": 0.10
    }
    
    # Thresholds
    THRESHOLD_HIGH = 0.80
    THRESHOLD_MEDIUM = 0.60
    THRESHOLD_LOW = 0.40
    
    # Decay factor for chain confidence
    DECAY_FACTOR = 0.95
    
    def __init__(self):
        """Initialize confidence scorer"""
        logger.info("Confidence scorer initialized")
    
    def calculate(
        self,
        evidence_quality: float,
        reasoning_coherence: float,
        llm_confidence: float,
        context_match: float
    ) -> float:
        """
        Calculate weighted confidence score
        
        Args:
            evidence_quality: Quality of supporting evidence (0-1)
            reasoning_coherence: Logical consistency (0-1)
            llm_confidence: LLM's self-reported confidence (0-1)
            context_match: RAG retrieval match score (0-1)
            
        Returns:
            Weighted confidence score (0-1)
        """
        # Validate inputs
        for value in [evidence_quality, reasoning_coherence, llm_confidence, context_match]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Confidence factor must be between 0 and 1, got {value}")
        
        # Weighted sum
        confidence = (
            evidence_quality * self.WEIGHTS["evidence_quality"] +
            reasoning_coherence * self.WEIGHTS["reasoning_coherence"] +
            llm_confidence * self.WEIGHTS["llm_confidence"] +
            context_match * self.WEIGHTS["context_match"]
        )
        
        logger.debug(
            f"Calculated confidence: {confidence:.3f} "
            f"(evidence: {evidence_quality:.2f}, coherence: {reasoning_coherence:.2f}, "
            f"llm: {llm_confidence:.2f}, context: {context_match:.2f})"
        )
        
        return confidence
    
    def calculate_from_factors(self, factors: ConfidenceFactors) -> float:
        """Calculate confidence from ConfidenceFactors dataclass"""
        return self.calculate(
            evidence_quality=factors.evidence_quality,
            reasoning_coherence=factors.reasoning_coherence,
            llm_confidence=factors.llm_confidence,
            context_match=factors.context_match
        )
    
    def aggregate_chain(self, confidences: List[float], use_decay: bool = True) -> float:
        """
        Aggregate confidence across a reasoning chain.
        Uses minimum of chain with optional decay for length.
        
        Args:
            confidences: List of confidence scores for each step
            use_decay: Apply decay factor for chain length
            
        Returns:
            Aggregated confidence
        """
        if not confidences:
            return 0.0
        
        # Minimum confidence in chain (weakest link)
        min_confidence = min(confidences)
        
        if use_decay:
            # Apply decay factor based on chain length
            # Longer chains have more uncertainty
            decay = self.DECAY_FACTOR ** len(confidences)
            aggregated = min_confidence * decay
        else:
            aggregated = min_confidence
        
        logger.debug(
            f"Chain aggregation: {aggregated:.3f} "
            f"(min: {min_confidence:.3f}, steps: {len(confidences)}, decay: {use_decay})"
        )
        
        return aggregated
    
    def aggregate_parallel(self, confidences: List[float]) -> float:
        """
        Aggregate confidence from parallel reasoning paths.
        Uses average (all paths contribute).
        
        Args:
            confidences: List of confidence scores from parallel paths
            
        Returns:
            Averaged confidence
        """
        if not confidences:
            return 0.0
        
        avg_confidence = sum(confidences) / len(confidences)
        
        logger.debug(
            f"Parallel aggregation: {avg_confidence:.3f} "
            f"(paths: {len(confidences)})"
        )
        
        return avg_confidence
    
    def get_level(self, confidence: float) -> str:
        """
        Get confidence level category
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence level string
        """
        if confidence >= self.THRESHOLD_HIGH:
            return ConfidenceLevel.HIGH
        elif confidence >= self.THRESHOLD_MEDIUM:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.THRESHOLD_LOW:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.INSUFFICIENT
    
    def should_proceed(self, confidence: float, threshold: Optional[float] = None) -> bool:
        """
        Determine if confidence is sufficient to proceed
        
        Args:
            confidence: Confidence score
            threshold: Custom threshold (default: THRESHOLD_LOW)
            
        Returns:
            True if confidence >= threshold
        """
        threshold = threshold or self.THRESHOLD_LOW
        return confidence >= threshold
    
    def calculate_evidence_quality(
        self,
        source_credibility: float,
        recency_score: float,
        sample_size_score: float
    ) -> float:
        """
        Calculate evidence quality factor
        
        Args:
            source_credibility: Credibility of source (0-1)
                - 1.0: Peer-reviewed journal
                - 0.7: Medical guideline
                - 0.5: Clinical notes
                - 0.3: Anecdotal
            recency_score: How recent the evidence is (0-1)
                - 1.0: < 1 year
                - 0.8: 1-3 years
                - 0.6: 3-5 years
                - 0.4: > 5 years
            sample_size_score: Quality of study (0-1)
                - 1.0: Large RCT
                - 0.7: Small RCT
                - 0.5: Observational
                - 0.3: Case study
                
        Returns:
            Evidence quality score (0-1)
        """
        # Weighted combination
        quality = (
            source_credibility * 0.5 +
            sample_size_score * 0.3 +
            recency_score * 0.2
        )
        
        return min(1.0, max(0.0, quality))
    
    def calculate_reasoning_coherence(
        self,
        has_contradictions: bool,
        is_complete_chain: bool,
        logical_consistency_score: float
    ) -> float:
        """
        Calculate reasoning coherence factor
        
        Args:
            has_contradictions: Whether reasoning has contradictions
            is_complete_chain: Whether causal chain is complete
            logical_consistency_score: Logical consistency (0-1)
            
        Returns:
            Reasoning coherence score (0-1)
        """
        if has_contradictions:
            coherence = 0.3  # Major penalty for contradictions
        elif not is_complete_chain:
            coherence = 0.6  # Moderate penalty for incomplete chain
        else:
            coherence = logical_consistency_score
        
        return coherence
    
    def explain(self, confidence: float, factors: Optional[ConfidenceFactors] = None) -> str:
        """
        Generate human-readable explanation of confidence score
        
        Args:
            confidence: Overall confidence score
            factors: Individual factors (optional)
            
        Returns:
            Explanation string
        """
        level = self.get_level(confidence)
        
        explanation = f"Confidence: {confidence:.1%} ({level})\n"
        
        if factors:
            explanation += "Factors:\n"
            explanation += f"  - Evidence Quality: {factors.evidence_quality:.1%}\n"
            explanation += f"  - Reasoning Coherence: {factors.reasoning_coherence:.1%}\n"
            explanation += f"  - LLM Confidence: {factors.llm_confidence:.1%}\n"
            explanation += f"  - Context Match: {factors.context_match:.1%}\n"
        
        if level == ConfidenceLevel.HIGH:
            explanation += "Recommendation: Proceed with confidence"
        elif level == ConfidenceLevel.MEDIUM:
            explanation += "Recommendation: Proceed with monitoring"
        elif level == ConfidenceLevel.LOW:
            explanation += "Recommendation: Human review recommended"
        else:
            explanation += "Recommendation: Expert consultation required"
        
        return explanation


# Singleton instance
_scorer_instance: Optional[ConfidenceScorer] = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create singleton ConfidenceScorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = ConfidenceScorer()
    return _scorer_instance
