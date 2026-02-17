"""Ollama embeddings integration."""

from typing import Optional

import ollama
from langchain_core.embeddings import Embeddings

from src.config import get_settings


class OllamaEmbeddings(Embeddings):
    """Ollama embeddings implementation compatible with LangChain."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """Initialize Ollama embeddings.

        Args:
            model: The embedding model name. Defaults to settings.
            base_url: The Ollama server URL. Defaults to settings.
            dimensions: The embedding dimensions. Defaults to settings.
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self.dimensions = dimensions or settings.embedding_dimensions

        self._client = ollama.Client(host=self.base_url)
        self._async_client = ollama.AsyncClient(host=self.base_url)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        response = self._client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            response = self._client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text asynchronously.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        response = await self._async_client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents asynchronously.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            response = await self._async_client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings
