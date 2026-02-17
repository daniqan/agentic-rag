"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google/Gemini
    google_api_key: str
    google_cloud_project: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash-preview-04-17"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-rag-demo"

    # Ollama Embeddings
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    # Firecrawl
    firecrawl_api_key: str

    # Zep Memory
    zep_api_key: str
    zep_api_url: str = "http://localhost:8000"

    # LangSmith
    langchain_api_key: str
    langchain_tracing_v2: bool = True
    langchain_project: str = "agentic-rag-demo"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
