"""Tests for Zep memory module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.zep_memory import ZepMemory


class TestZepMemory:
    """Test ZepMemory class."""

    def test_init_with_defaults(self, mock_settings):
        """ZepMemory should initialize with default settings."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                memory = ZepMemory(session_id="test-session")

                assert memory.session_id == "test-session"
                mock_zep.assert_called_once_with(
                    api_key=mock_settings.zep_api_key,
                    api_url=mock_settings.zep_api_url,
                )

    def test_init_with_custom_values(self, mock_settings):
        """ZepMemory should accept custom configuration."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                memory = ZepMemory(
                    session_id="custom-session",
                    api_key="custom-key",
                    api_url="http://custom:8000",
                )

                assert memory.session_id == "custom-session"
                mock_zep.assert_called_once_with(
                    api_key="custom-key",
                    api_url="http://custom:8000",
                )

    def test_add_message(self, mock_settings):
        """add_message should store a message in the session."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                memory.add_message(role="user", content="Hello")

                mock_client.memory.add.assert_called_once()

    def test_add_user_message(self, mock_settings):
        """add_user_message should add a user message."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                memory.add_user_message("Hello, how are you?")

                mock_client.memory.add.assert_called_once()
                call_args = mock_client.memory.add.call_args
                assert call_args[0][0] == "test-session"

    def test_add_ai_message(self, mock_settings):
        """add_ai_message should add an AI message."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                memory.add_ai_message("I'm doing well, thank you!")

                mock_client.memory.add.assert_called_once()

    def test_get_messages(self, mock_settings):
        """get_messages should return conversation history."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_memory = MagicMock()
                mock_memory.messages = [
                    MagicMock(role="user", content="Hello"),
                    MagicMock(role="assistant", content="Hi there!"),
                ]
                mock_client.memory.get.return_value = mock_memory
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                messages = memory.get_messages()

                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[0]["content"] == "Hello"

    def test_get_messages_empty(self, mock_settings):
        """get_messages should return empty list for new session."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_memory = MagicMock()
                mock_memory.messages = []
                mock_client.memory.get.return_value = mock_memory
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                messages = memory.get_messages()

                assert messages == []

    def test_clear(self, mock_settings):
        """clear should delete the session memory."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                memory.clear()

                mock_client.memory.delete.assert_called_once_with("test-session")

    def test_search(self, mock_settings):
        """search should find relevant memories."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_result = MagicMock()
                mock_result.results = [
                    MagicMock(message=MagicMock(content="Found message"), score=0.9),
                ]
                mock_client.memory.search.return_value = mock_result
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                results = memory.search("query", limit=5)

                assert len(results) == 1
                assert results[0]["content"] == "Found message"
                assert results[0]["score"] == 0.9

    def test_get_summary(self, mock_settings):
        """get_summary should return conversation summary."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient") as mock_zep:
                mock_client = MagicMock()
                mock_memory = MagicMock()
                mock_memory.summary = MagicMock(content="Summary of conversation")
                mock_client.memory.get.return_value = mock_memory
                mock_zep.return_value = mock_client

                memory = ZepMemory(session_id="test-session")
                summary = memory.get_summary()

                assert summary == "Summary of conversation"

    def test_as_langchain_memory(self, mock_settings):
        """as_langchain_memory should return LangChain compatible memory."""
        with patch("src.memory.zep_memory.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.memory.zep_memory.ZepClient"):
                memory = ZepMemory(session_id="test-session")
                lc_memory = memory.as_langchain_memory()

                assert lc_memory is not None
