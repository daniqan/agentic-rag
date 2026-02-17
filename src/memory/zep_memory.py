"""Zep memory integration for conversation history."""

from typing import Any, Optional

from zep_cloud.client import Zep as ZepClient
from zep_cloud.types import Message

from src.config import get_settings


class ZepMemory:
    """Zep memory manager for conversation history."""

    def __init__(
        self,
        session_id: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """Initialize Zep memory.

        Args:
            session_id: The session identifier.
            api_key: Optional API key. Defaults to settings.
            api_url: Optional API URL. Defaults to settings.
        """
        settings = get_settings()
        self.session_id = session_id

        self._client = ZepClient(
            api_key=api_key or settings.zep_api_key,
            api_url=api_url or settings.zep_api_url,
        )

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session.

        Args:
            role: The role (user, assistant, system).
            content: The message content.
        """
        message = Message(role=role, content=content)
        self._client.memory.add(
            self.session_id,
            messages=[message],
        )

    def add_user_message(self, content: str) -> None:
        """Add a user message.

        Args:
            content: The message content.
        """
        self.add_message("user", content)

    def add_ai_message(self, content: str) -> None:
        """Add an AI/assistant message.

        Args:
            content: The message content.
        """
        self.add_message("assistant", content)

    def get_messages(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get conversation messages.

        Args:
            limit: Maximum number of messages to retrieve.

        Returns:
            List of message dictionaries.
        """
        memory = self._client.memory.get(self.session_id)

        if not memory or not memory.messages:
            return []

        messages = []
        for msg in memory.messages[:limit]:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        return messages

    def clear(self) -> None:
        """Clear the session memory."""
        self._client.memory.delete(self.session_id)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory for relevant messages.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of search results with content and score.
        """
        result = self._client.memory.search(
            self.session_id,
            text=query,
            limit=limit,
        )

        results = []
        for r in result.results:
            results.append({
                "content": r.message.content,
                "score": r.score,
            })
        return results

    def get_summary(self) -> Optional[str]:
        """Get conversation summary.

        Returns:
            The conversation summary or None.
        """
        memory = self._client.memory.get(self.session_id)

        if not memory or not memory.summary:
            return None

        return memory.summary.content

    def as_langchain_memory(self) -> "ZepLangChainMemory":
        """Get a LangChain compatible memory wrapper.

        Returns:
            A LangChain memory wrapper.
        """
        return ZepLangChainMemory(self)


class ZepLangChainMemory:
    """LangChain compatible wrapper for ZepMemory."""

    def __init__(self, zep_memory: ZepMemory):
        """Initialize the wrapper.

        Args:
            zep_memory: The ZepMemory instance.
        """
        self._memory = zep_memory
        self.memory_key = "history"
        self.input_key = "input"
        self.output_key = "output"

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory variables.

        Args:
            inputs: Input variables (unused).

        Returns:
            Dictionary with history.
        """
        messages = self._memory.get_messages()
        return {self.memory_key: messages}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save context from conversation.

        Args:
            inputs: Input dictionary.
            outputs: Output dictionary.
        """
        if self.input_key in inputs:
            self._memory.add_user_message(inputs[self.input_key])
        if self.output_key in outputs:
            self._memory.add_ai_message(outputs[self.output_key])

    def clear(self) -> None:
        """Clear memory."""
        self._memory.clear()
