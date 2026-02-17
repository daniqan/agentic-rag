"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_env():
    """Fixture providing mock environment variables for testing."""
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
        yield env_vars


@pytest.fixture
def mock_settings(mock_env):
    """Fixture providing a Settings instance with mock values."""
    from src.config import Settings, get_settings

    get_settings.cache_clear()
    return Settings()


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            "page_content": "Python is a programming language.",
            "metadata": {"source": "test.pdf", "page": 1},
        },
        {
            "page_content": "Machine learning is a subset of AI.",
            "metadata": {"source": "test.pdf", "page": 2},
        },
        {
            "page_content": "LangChain is a framework for LLM applications.",
            "metadata": {"source": "https://example.com", "type": "web"},
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Fixture providing sample embeddings for testing (768 dimensions)."""
    import random

    random.seed(42)
    return [[random.random() for _ in range(768)] for _ in range(3)]


@pytest.fixture
def mock_ollama_client():
    """Fixture providing a mock Ollama client."""
    mock = MagicMock()
    mock.embeddings.return_value = {"embedding": [0.1] * 768}
    return mock


@pytest.fixture
def mock_pinecone_index():
    """Fixture providing a mock Pinecone index."""
    mock = MagicMock()
    mock.upsert.return_value = {"upserted_count": 1}
    mock.query.return_value = MagicMock(
        matches=[
            MagicMock(id="1", score=0.9, metadata={"text": "test content"}),
        ]
    )
    return mock


@pytest.fixture
def mock_gemini_response():
    """Fixture providing a mock Gemini response."""
    mock = MagicMock()
    mock.content = "This is a test response from Gemini."
    return mock
