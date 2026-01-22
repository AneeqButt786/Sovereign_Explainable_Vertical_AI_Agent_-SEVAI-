"""
Configuration management for SEVAI
Load and validate environment variables
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4"
    
    # Anthropic Configuration (Optional)
    anthropic_api_key: Optional[str] = None
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "sevai-medical-knowledge"
    
    # Database Configuration
    database_url: str = "sqlite:///./sevai_vault.db"
    
    # Application Settings
    app_env: str = "development"
    log_level: str = "INFO"
    max_retries: int = 3
    request_timeout: int = 30
    
    # Vector Store Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_vectors: int = 95000
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Confidence Thresholds
    confidence_threshold_high: float = 0.85
    confidence_threshold_medium: float = 0.70
    confidence_threshold_low: float = 0.50
    
    # HIPAA Compliance
    enable_phi_detection: bool = True
    enable_audit_logging: bool = True
    audit_log_path: str = "./logs/audit.log"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


def get_settings() -> Settings:
    """Get application settings singleton"""
    return Settings()
