"""Main RAG Agent implementation."""

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from src.config import get_settings
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.guardrails.validators import validate_input, validate_output
from src.llm.gemini import GeminiLLM
from src.retrieval.pinecone_store import PineconeStore


@dataclass
class RAGAgentConfig:
    """Configuration for the RAG Agent."""

    k: int = 4
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    validate_input: bool = True
    validate_output: bool = True
    use_memory: bool = True
    system_prompt: str = field(
        default="""You are a helpful AI assistant that answers questions based on the provided context.
Use the following context to answer the user's question.
If you don't know the answer based on the context, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
    )


class RAGAgent:
    """Agentic RAG implementation with LangChain."""

    def __init__(
        self,
        config: Optional[RAGAgentConfig] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        llm: Optional[GeminiLLM] = None,
        vector_store: Optional[PineconeStore] = None,
    ):
        """Initialize the RAG Agent.

        Args:
            config: Agent configuration.
            embeddings: Optional embeddings model.
            llm: Optional LLM instance.
            vector_store: Optional vector store.
        """
        self.config = config or RAGAgentConfig()
        settings = get_settings()

        # Initialize embeddings
        self.embeddings = embeddings or OllamaEmbeddings()

        # Initialize LLM
        self.llm = llm or GeminiLLM(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Initialize vector store
        self.vector_store = vector_store or PineconeStore(
            embeddings=self.embeddings,
        )

        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_template(self.config.system_prompt)

    def query(self, question: str) -> str:
        """Query the RAG system.

        Args:
            question: The user's question.

        Returns:
            The generated response.

        Raises:
            ValueError: If input validation fails.
        """
        # Validate input
        if self.config.validate_input:
            validation = validate_input(question)
            if not validation.is_valid:
                raise ValueError(f"Invalid input: {', '.join(validation.errors)}")

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=self.config.k)

        # Build context from documents
        context = self._build_context(docs)

        # Generate response
        prompt_value = self.prompt.format_messages(
            context=context,
            question=question,
        )
        response = self.llm.invoke(prompt_value)

        # Validate output
        if self.config.validate_output:
            validation = validate_output(response)
            if not validation.is_valid:
                response = self._handle_invalid_output(response, validation.errors)

        return response

    async def aquery(self, question: str) -> str:
        """Query the RAG system asynchronously.

        Args:
            question: The user's question.

        Returns:
            The generated response.

        Raises:
            ValueError: If input validation fails.
        """
        # Validate input
        if self.config.validate_input:
            validation = validate_input(question)
            if not validation.is_valid:
                raise ValueError(f"Invalid input: {', '.join(validation.errors)}")

        # Retrieve relevant documents
        docs = await self.vector_store.asimilarity_search(question, k=self.config.k)

        # Build context from documents
        context = self._build_context(docs)

        # Generate response
        prompt_value = self.prompt.format_messages(
            context=context,
            question=question,
        )
        response = await self.llm.ainvoke(prompt_value)

        # Validate output
        if self.config.validate_output:
            validation = validate_output(response)
            if not validation.is_valid:
                response = self._handle_invalid_output(response, validation.errors)

        return response

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            List of document IDs.
        """
        return self.vector_store.add_documents(documents)

    def get_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Get a retriever for the vector store.

        Args:
            **kwargs: Arguments passed to the retriever.

        Returns:
            A LangChain retriever.
        """
        search_kwargs = kwargs.pop("search_kwargs", {"k": self.config.k})
        return self.vector_store.as_retriever(search_kwargs=search_kwargs, **kwargs)

    def _build_context(self, docs: list[Document]) -> str:
        """Build context string from documents.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        if not docs:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] Source: {source}\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def _handle_invalid_output(self, response: str, errors: list[str]) -> str:
        """Handle invalid output from the model.

        Args:
            response: The original response.
            errors: List of validation errors.

        Returns:
            A sanitized response.
        """
        return (
            "I apologize, but I cannot provide that response as it may contain "
            "inappropriate content. Please try rephrasing your question."
        )
