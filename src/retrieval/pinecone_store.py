"""Pinecone vector store integration."""

import uuid
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone

from src.config import get_settings


class PineconeStore(VectorStore):
    """Pinecone vector store implementation compatible with LangChain."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        text_key: str = "text",
        namespace: Optional[str] = None,
    ):
        """Initialize Pinecone store.

        Args:
            index_name: The Pinecone index name. Defaults to settings.
            embeddings: The embeddings model to use.
            text_key: The metadata key for document text.
            namespace: Optional namespace for the index.
        """
        settings = get_settings()
        self.index_name = index_name or settings.pinecone_index_name
        self._embeddings = embeddings
        self.text_key = text_key
        self.namespace = namespace

        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._client.Index(self.index_name)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Return the embeddings model."""
        return self._embeddings

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: List of texts to add.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs.

        Returns:
            List of IDs for the added texts.
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model required for add_texts")

        embeddings = self._embeddings.embed_documents(list(texts))

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        vectors = []
        for i, (text, embedding, metadata) in enumerate(
            zip(texts, embeddings, metadatas)
        ):
            vector_metadata = {**metadata, self.text_key: text}
            vectors.append({"id": ids[i], "values": embedding, "metadata": vector_metadata})

        self._index.upsert(vectors=vectors, namespace=self.namespace)
        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
            ids: Optional list of IDs.

        Returns:
            List of IDs for the added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            List of similar documents.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            List of (document, score) tuples.
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model required for similarity search")

        query_embedding = self._embeddings.embed_query(query)

        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )

        docs_and_scores = []
        for match in results.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self.text_key, "")
            doc = Document(page_content=text, metadata=metadata)
            docs_and_scores.append((doc, match.score))

        return docs_and_scores

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Async search for similar documents.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            List of similar documents.
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model required for similarity search")

        query_embedding = await self._embeddings.aembed_query(query)

        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )

        docs = []
        for match in results.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self.text_key, "")
            doc = Document(page_content=text, metadata=metadata)
            docs.append(doc)

        return docs

    def delete(
        self,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self._index.delete(ids=ids, namespace=self.namespace, **kwargs)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever for this vector store.

        Args:
            **kwargs: Keyword arguments passed to the retriever.

        Returns:
            A LangChain retriever.
        """
        from langchain_core.vectorstores import VectorStoreRetriever

        search_kwargs = kwargs.pop("search_kwargs", {})
        return VectorStoreRetriever(
            vectorstore=self, search_kwargs=search_kwargs, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "PineconeStore":
        """Create a PineconeStore from texts.

        Args:
            texts: List of texts to add.
            embedding: Embeddings model to use.
            metadatas: Optional list of metadata dicts.

        Returns:
            A new PineconeStore instance.
        """
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store
