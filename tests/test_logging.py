"""
Test configuration and logging
"""

import pytest
from core.logging_config import setup_logging, get_logger
from pathlib import Path
import tempfile


def test_logging_setup():
    """Test basic logging configuration"""
    logger = setup_logging(log_level="DEBUG", log_file=Path(tempfile.gettempdir()) / "test.log")
    assert logger is not None
    assert logger.name == "sevai"


def test_get_logger():
    """Test get_logger function"""
    logger = get_logger("test_module")
    assert logger.name == "sevai.test_module"


def test_audit_logger():
    """Test audit logger exists"""
    from core.logging_config import get_audit_logger
    audit_logger = get_audit_logger()
    assert audit_logger.name == "sevai.audit"
