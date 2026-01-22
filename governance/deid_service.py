"""
De-identification Service - PII/PHI masking for medical records
"""

import re
from typing import List, Dict, Any, Optional
from core.logging_config import get_logger

logger = get_logger("deid_service")

class DeidService:
    """
    Detects and masks Protected Health Information (PHI) and 
    Personally Identifiable Information (PII) in text and logs.
    """
    
    def __init__(self):
        # Basic patterns for common PHI entities
        self.patterns = {
            "MRN": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{7,10}\b", # Medical Record Number
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "DOB": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "ZIP": r"\b\d{5}(?:-\d{4})?\b"
        }
        logger.info("De-identification Service initialized")

    def mask_text(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Scan text for PHI patterns and mask them
        """
        masked_text = text
        for label, pattern in self.patterns.items():
            matches = re.findall(pattern, masked_text)
            if matches:
                logger.debug(f"Detected {len(matches)} occurrences of {label}")
                masked_text = re.sub(pattern, f"{replacement} (PHI:{label})", masked_text)
        
        return masked_text

    def mask_dict(self, data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Recursively mask values in a dictionary
        """
        masked_dict = {}
        target_keys = sensitive_keys or ["name", "address", "dob", "patient_id", "mrn", "content"]
        
        for k, v in data.items():
            if isinstance(v, str):
                if k.lower() in target_keys:
                    masked_dict[k] = self.mask_text(v)
                else:
                    masked_dict[k] = v
            elif isinstance(v, dict):
                masked_dict[k] = self.mask_dict(v, target_keys)
            elif isinstance(v, list):
                masked_dict[k] = [
                    self.mask_dict(item, target_keys) if isinstance(item, dict) else item 
                    for item in v
                ]
            else:
                masked_dict[k] = v
                
        return masked_dict

# Singleton
_deid_instance: Optional[DeidService] = None

def get_deid_service() -> DeidService:
    global _deid_instance
    if _deid_instance is None:
        _deid_instance = DeidService()
    return _deid_instance
