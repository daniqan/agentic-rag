"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration class."""

    def test_settings_loads_from_env(self):
        """Settings should load values from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test-google-key",
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "PINECONE_API_KEY": "test-pinecone-key",
            "PINECONE_INDEX_NAME": "test-index",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "FIRECRAWL_API_KEY": "test-firecrawl-key",
            "ZEP_API_KEY": "test-zep-key",
            "ZEP_API_URL": "http://localhost:8000",
            "LANGCHAIN_API_KEY": "test-langchain-key",
            "LANGCHAIN_PROJECT": "test-project",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert settings.google_api_key == "test-google-key"
            assert settings.google_cloud_project == "test-project"
            assert settings.pinecone_api_key == "test-pinecone-key"
            assert settings.pinecone_index_name == "test-index"
            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.firecrawl_api_key == "test-firecrawl-key"
            assert settings.zep_api_key == "test-zep-key"
            assert settings.zep_api_url == "http://localhost:8000"
            assert settings.langchain_api_key == "test-langchain-key"
            assert settings.langchain_project == "test-project"

    def test_settings_has_defaults(self):
        """Settings should have sensible defaults for optional values."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "PINECONE_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "test-key",
            "ZEP_API_KEY": "test-key",
            "LANGCHAIN_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.pinecone_index_name == "agentic-rag-demo"
            assert settings.zep_api_url == "http://localhost:8000"
            assert settings.langchain_project == "agentic-rag-demo"
            assert settings.langchain_tracing_v2 is True

    def test_settings_embedding_model_default(self):
        """Settings should have default embedding model."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "PINECONE_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "test-key",
            "ZEP_API_KEY": "test-key",
            "LANGCHAIN_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            assert settings.embedding_model == "nomic-embed-text"
            assert settings.embedding_dimensions == 768

    def test_settings_gemini_model_default(self):
        """Settings should have default Gemini model."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "PINECONE_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "test-key",
            "ZEP_API_KEY": "test-key",
            "LANGCHAIN_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            assert settings.gemini_model == "gemini-2.5-flash-preview-04-17"


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """get_settings should return a Settings instance."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "PINECONE_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "test-key",
            "ZEP_API_KEY": "test-key",
            "LANGCHAIN_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_get_settings_caches_instance(self):
        """get_settings should return the same cached instance."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "PINECONE_API_KEY": "test-key",
            "FIRECRAWL_API_KEY": "test-key",
            "ZEP_API_KEY": "test-key",
            "LANGCHAIN_API_KEY": "test-key",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear cache first
            get_settings.cache_clear()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2
