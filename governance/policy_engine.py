"""
Policy Engine - Rule-based compliance checking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from core.logging_config import get_logger

logger = get_logger("policy_engine")

class PolicyAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    ESCALATE = "escalate"

@dataclass
class Rule:
    """A governance rule definition"""
    id: str
    description: str
    priority: int  # Higher is more critical
    action: PolicyAction
    condition_type: str  # e.g., "contains_pii", "confidence_threshold", "keyword_match"
    parameters: Dict[str, Any]

class PolicyEngine:
    """
    Evaluates agent outputs and system actions against regulatory policies.
    Supports HIPAA, GDPR, and custom institutional rules.
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or self._load_default_rules()
        logger.info(f"Policy Engine initialized with {len(self.rules)} rules")

    def _load_default_rules(self) -> List[Rule]:
        """Load default HIPAA and safety rules"""
        return [
            Rule(
                id="HIPAA-001",
                description="Prevent unmasked PHI in final output",
                priority=100,
                action=PolicyAction.BLOCK,
                condition_type="contains_phi",
                parameters={}
            ),
            Rule(
                id="SAFETY-001",
                description="Minimum confidence for treatment suggestions",
                priority=90,
                action=PolicyAction.WARN,
                condition_type="confidence_threshold",
                parameters={"threshold": 0.7}
            ),
            Rule(
                id="SAFETY-002",
                description="Halt on critical contradictions in diagnosis",
                priority=95,
                action=PolicyAction.ESCALATE,
                condition_type="critical_contradiction",
                parameters={}
            )
        ]

    def evaluate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate data against all active rules
        
        Returns:
            List of violations/findings
        """
        findings = []
        for rule in sorted(self.rules, key=lambda x: x.priority, reverse=True):
            logger.debug(f"Evaluating rule: {rule.id}")
            # Logic for specific condition types will be implemented in Step 2
            # For now, this is a stub for the structure
        return findings

# Singleton instance
_engine_instance: Optional[PolicyEngine] = None

def get_policy_engine() -> PolicyEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PolicyEngine()
    return _engine_instance
