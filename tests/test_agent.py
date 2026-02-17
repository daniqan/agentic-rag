"""Tests for RAG agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.rag_agent import RAGAgent, RAGAgentConfig


class TestRAGAgentConfig:
    """Test RAGAgentConfig dataclass."""

    def test_default_config(self):
        """RAGAgentConfig should have sensible defaults."""
        config = RAGAgentConfig()

        assert config.k == 4
        assert config.temperature == 0.7
        assert config.validate_input is True
        assert config.validate_output is True
        assert config.use_memory is True

    def test_custom_config(self):
        """RAGAgentConfig should accept custom values."""
        config = RAGAgentConfig(
            k=10,
            temperature=0.5,
            validate_input=False,
        )

        assert config.k == 10
        assert config.temperature == 0.5
        assert config.validate_input is False


class TestRAGAgent:
    """Test RAGAgent class."""

    def test_init(self, mock_settings):
        """RAGAgent should initialize with required components."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        agent = RAGAgent()

                        assert agent.llm is not None
                        assert agent.vector_store is not None

    def test_init_with_config(self, mock_settings):
        """RAGAgent should accept custom config."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        config = RAGAgentConfig(k=10, temperature=0.5)
                        agent = RAGAgent(config=config)

                        assert agent.config.k == 10
                        assert agent.config.temperature == 0.5

    def test_query(self, mock_settings):
        """query should return response based on retrieved context."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        mock_llm = MagicMock()
                        mock_llm.invoke.return_value = "This is the answer."
                        mock_llm_class.return_value = mock_llm

                        mock_store = MagicMock()
                        mock_store.similarity_search.return_value = [
                            Document(
                                page_content="Relevant content",
                                metadata={"source": "test"},
                            )
                        ]
                        mock_store_class.return_value = mock_store

                        agent = RAGAgent()
                        response = agent.query("What is Python?")

                        assert "answer" in response.lower()
                        mock_store.similarity_search.assert_called_once()
                        mock_llm.invoke.assert_called_once()

    def test_query_with_validation(self, mock_settings):
        """query should validate input when enabled."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        with patch("src.agents.rag_agent.validate_input") as mock_validate:
                            mock_validate.return_value = MagicMock(
                                is_valid=True, value="test query", errors=[]
                            )

                            mock_llm = MagicMock()
                            mock_llm.invoke.return_value = "Response"
                            mock_llm_class.return_value = mock_llm

                            mock_store = MagicMock()
                            mock_store.similarity_search.return_value = []
                            mock_store_class.return_value = mock_store

                            agent = RAGAgent()
                            agent.query("test query")

                            mock_validate.assert_called_once()

    def test_query_rejects_invalid_input(self, mock_settings):
        """query should reject invalid input."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        agent = RAGAgent()

                        with pytest.raises(ValueError):
                            agent.query("")

    @pytest.mark.asyncio
    async def test_aquery(self, mock_settings):
        """aquery should return response asynchronously."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        mock_llm = MagicMock()
                        mock_llm.ainvoke = AsyncMock(return_value="Async answer")
                        mock_llm_class.return_value = mock_llm

                        mock_store = MagicMock()
                        mock_store.asimilarity_search = AsyncMock(return_value=[])
                        mock_store_class.return_value = mock_store

                        agent = RAGAgent()
                        response = await agent.aquery("Test question")

                        assert response == "Async answer"

    def test_add_documents(self, mock_settings):
        """add_documents should store documents in vector store."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        mock_store = MagicMock()
                        mock_store.add_documents.return_value = ["id1", "id2"]
                        mock_store_class.return_value = mock_store

                        agent = RAGAgent()
                        docs = [
                            Document(page_content="Doc 1"),
                            Document(page_content="Doc 2"),
                        ]
                        ids = agent.add_documents(docs)

                        assert len(ids) == 2
                        mock_store.add_documents.assert_called_once()

    def test_get_retriever(self, mock_settings):
        """get_retriever should return a retriever."""
        with patch("src.agents.rag_agent.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.agents.rag_agent.GeminiLLM") as mock_llm_class:
                with patch("src.agents.rag_agent.PineconeStore") as mock_store_class:
                    with patch("src.agents.rag_agent.OllamaEmbeddings") as mock_embed:
                        mock_store = MagicMock()
                        mock_retriever = MagicMock()
                        mock_store.as_retriever.return_value = mock_retriever
                        mock_store_class.return_value = mock_store

                        agent = RAGAgent()
                        retriever = agent.get_retriever()

                        assert retriever is mock_retriever
