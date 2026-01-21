"""
Logging configuration for SEVAI
Provides structured logging with HIPAA-compliant audit trails
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import os


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_audit: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for SEVAI
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to env var LOG_LEVEL
        log_file: Path to log file. If None, logs to stdout only
        enable_audit: Whether to enable separate audit logging
        
    Returns:
        Configured logger instance
    """
    # Get log level from env or parameter
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("sevai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (rotating)
    if log_file is None:
        log_file = log_dir / "sevai.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Audit logger (HIPAA compliance)
    if enable_audit:
        audit_logger = logging.getLogger("sevai.audit")
        audit_logger.setLevel(logging.INFO)
        
        audit_file = log_dir / "audit.log"
        audit_handler = RotatingFileHandler(
            audit_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB (larger for audit logs)
            backupCount=10
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False  # Don't propagate to root logger
    
    logger.info(f"Logging initialized at {log_level} level")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(f"sevai.{name}")


def get_audit_logger() -> logging.Logger:
    """Get the audit logger for HIPAA-compliant logging"""
    return logging.getLogger("sevai.audit")
