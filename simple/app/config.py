# app/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/procurement"
    
    # Vector DB (for SQL agent)
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    
    # Application
    APP_NAME: str = "Procurement Multi-Agent System"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Agent Configuration
    MODEL_NAME: str = "claude-3-5-sonnet-20241022"
    MAX_ITERATIONS: int = 15  # Prevent infinite loops
    TEMPERATURE: float = 0.0
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()