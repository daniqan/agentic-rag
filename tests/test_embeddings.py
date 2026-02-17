"""Tests for Ollama embeddings module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.embeddings.ollama_embeddings import OllamaEmbeddings


class TestOllamaEmbeddings:
    """Test OllamaEmbeddings class."""

    def test_init_with_defaults(self, mock_settings):
        """OllamaEmbeddings should initialize with default settings."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            embeddings = OllamaEmbeddings()

            assert embeddings.model == mock_settings.embedding_model
            assert embeddings.base_url == mock_settings.ollama_base_url
            assert embeddings.dimensions == mock_settings.embedding_dimensions

    def test_init_with_custom_values(self, mock_settings):
        """OllamaEmbeddings should accept custom configuration."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            embeddings = OllamaEmbeddings(
                model="custom-model",
                base_url="http://custom:11434",
                dimensions=512,
            )

            assert embeddings.model == "custom-model"
            assert embeddings.base_url == "http://custom:11434"
            assert embeddings.dimensions == 512

    def test_embed_query(self, mock_settings):
        """embed_query should return embedding for a single query."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.embeddings.ollama_embeddings.ollama") as mock_ollama:
                mock_ollama.Client.return_value.embeddings.return_value = {
                    "embedding": [0.1] * 768
                }

                embeddings = OllamaEmbeddings()
                result = embeddings.embed_query("test query")

                assert len(result) == 768
                assert all(isinstance(x, float) for x in result)
                mock_ollama.Client.return_value.embeddings.assert_called_once()

    def test_embed_documents(self, mock_settings):
        """embed_documents should return embeddings for multiple documents."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.embeddings.ollama_embeddings.ollama") as mock_ollama:
                mock_ollama.Client.return_value.embeddings.return_value = {
                    "embedding": [0.1] * 768
                }

                embeddings = OllamaEmbeddings()
                docs = ["doc1", "doc2", "doc3"]
                result = embeddings.embed_documents(docs)

                assert len(result) == 3
                assert all(len(emb) == 768 for emb in result)

    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_settings):
        """aembed_query should return embedding asynchronously."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.embeddings.ollama_embeddings.ollama") as mock_ollama:
                mock_client = AsyncMock()
                mock_client.embeddings.return_value = {"embedding": [0.1] * 768}
                mock_ollama.AsyncClient.return_value = mock_client

                embeddings = OllamaEmbeddings()
                result = await embeddings.aembed_query("test query")

                assert len(result) == 768

    @pytest.mark.asyncio
    async def test_aembed_documents(self, mock_settings):
        """aembed_documents should return embeddings asynchronously."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.embeddings.ollama_embeddings.ollama") as mock_ollama:
                mock_client = AsyncMock()
                mock_client.embeddings.return_value = {"embedding": [0.1] * 768}
                mock_ollama.AsyncClient.return_value = mock_client

                embeddings = OllamaEmbeddings()
                docs = ["doc1", "doc2"]
                result = await embeddings.aembed_documents(docs)

                assert len(result) == 2
                assert all(len(emb) == 768 for emb in result)

    def test_langchain_compatible(self, mock_settings):
        """OllamaEmbeddings should be compatible with LangChain Embeddings interface."""
        with patch("src.embeddings.ollama_embeddings.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            from langchain_core.embeddings import Embeddings

            embeddings = OllamaEmbeddings()
            assert isinstance(embeddings, Embeddings)
