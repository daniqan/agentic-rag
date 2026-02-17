"""Tests for Pinecone vector store module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.pinecone_store import PineconeStore


class TestPineconeStore:
    """Test PineconeStore class."""

    def test_init_with_defaults(self, mock_settings):
        """PineconeStore should initialize with default settings."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_pc.return_value.Index.return_value = MagicMock()

                store = PineconeStore()

                assert store.index_name == mock_settings.pinecone_index_name
                mock_pc.assert_called_once_with(api_key=mock_settings.pinecone_api_key)

    def test_init_with_custom_index(self, mock_settings):
        """PineconeStore should accept custom index name."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_pc.return_value.Index.return_value = MagicMock()

                store = PineconeStore(index_name="custom-index")

                assert store.index_name == "custom-index"

    def test_add_documents(self, mock_settings, sample_embeddings):
        """add_documents should store documents with embeddings."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_index = MagicMock()
                mock_index.upsert.return_value = {"upserted_count": 2}
                mock_pc.return_value.Index.return_value = mock_index

                mock_embeddings = MagicMock()
                mock_embeddings.embed_documents.return_value = sample_embeddings[:2]

                store = PineconeStore(embeddings=mock_embeddings)

                docs = [
                    Document(page_content="Doc 1", metadata={"source": "test1"}),
                    Document(page_content="Doc 2", metadata={"source": "test2"}),
                ]

                ids = store.add_documents(docs)

                assert len(ids) == 2
                mock_index.upsert.assert_called_once()

    def test_similarity_search(self, mock_settings, sample_embeddings):
        """similarity_search should return relevant documents."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_index = MagicMock()
                mock_index.query.return_value = MagicMock(
                    matches=[
                        MagicMock(
                            id="1",
                            score=0.95,
                            metadata={"text": "Doc 1", "source": "test1"},
                        ),
                        MagicMock(
                            id="2",
                            score=0.85,
                            metadata={"text": "Doc 2", "source": "test2"},
                        ),
                    ]
                )
                mock_pc.return_value.Index.return_value = mock_index

                mock_embeddings = MagicMock()
                mock_embeddings.embed_query.return_value = sample_embeddings[0]

                store = PineconeStore(embeddings=mock_embeddings)
                results = store.similarity_search("test query", k=2)

                assert len(results) == 2
                assert isinstance(results[0], Document)
                mock_index.query.assert_called_once()

    def test_similarity_search_with_score(self, mock_settings, sample_embeddings):
        """similarity_search_with_score should return documents with scores."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_index = MagicMock()
                mock_index.query.return_value = MagicMock(
                    matches=[
                        MagicMock(
                            id="1",
                            score=0.95,
                            metadata={"text": "Doc 1", "source": "test1"},
                        ),
                    ]
                )
                mock_pc.return_value.Index.return_value = mock_index

                mock_embeddings = MagicMock()
                mock_embeddings.embed_query.return_value = sample_embeddings[0]

                store = PineconeStore(embeddings=mock_embeddings)
                results = store.similarity_search_with_score("test query", k=1)

                assert len(results) == 1
                doc, score = results[0]
                assert isinstance(doc, Document)
                assert isinstance(score, float)
                assert score == 0.95

    def test_delete_documents(self, mock_settings):
        """delete should remove documents by IDs."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_index = MagicMock()
                mock_pc.return_value.Index.return_value = mock_index

                store = PineconeStore()
                store.delete(["id1", "id2"])

                mock_index.delete.assert_called_once_with(ids=["id1", "id2"], namespace=None)

    @pytest.mark.asyncio
    async def test_asimilarity_search(self, mock_settings, sample_embeddings):
        """asimilarity_search should return documents asynchronously."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_index = MagicMock()
                mock_index.query.return_value = MagicMock(
                    matches=[
                        MagicMock(
                            id="1",
                            score=0.95,
                            metadata={"text": "Doc 1", "source": "test1"},
                        ),
                    ]
                )
                mock_pc.return_value.Index.return_value = mock_index

                mock_embeddings = MagicMock()
                mock_embeddings.aembed_query = AsyncMock(
                    return_value=sample_embeddings[0]
                )

                store = PineconeStore(embeddings=mock_embeddings)
                results = await store.asimilarity_search("test query", k=1)

                assert len(results) == 1
                assert isinstance(results[0], Document)

    def test_as_retriever(self, mock_settings):
        """as_retriever should return a LangChain retriever."""
        with patch("src.retrieval.pinecone_store.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.retrieval.pinecone_store.Pinecone") as mock_pc:
                mock_pc.return_value.Index.return_value = MagicMock()

                store = PineconeStore()
                retriever = store.as_retriever(search_kwargs={"k": 5})

                assert retriever is not None
                assert retriever.search_kwargs.get("k") == 5
