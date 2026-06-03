# backend/config.py
# ============================================================
# Centralized configuration using Pydantic Settings
# All environment variables are loaded here once
# ============================================================

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic automatically reads from .env file.
    """

    # App metadata
    app_name: str = "Enterprise AI Knowledge Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # Backend server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Vector store
    vector_store_path: str = "./data/vectorstore"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding model (runs locally, no API needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM settings
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.3
    max_tokens: int = 2048

    # RAG retrieval
    top_k_results: int = 5

    # Simple auth
    auth_username: str = "admin"
    auth_password: str = "admin123"
    secret_key: str = "change-me-in-production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached settings instance.
    lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Create directories on startup
def ensure_directories():
    settings = get_settings()
    os.makedirs(settings.vector_store_path, exist_ok=True)
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
