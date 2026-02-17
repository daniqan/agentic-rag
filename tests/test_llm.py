"""Tests for Gemini LLM module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.gemini import GeminiLLM


class TestGeminiLLM:
    """Test GeminiLLM class."""

    def test_init_with_defaults(self, mock_settings):
        """GeminiLLM should initialize with default settings."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                llm = GeminiLLM()

                assert llm.model_name == mock_settings.gemini_model
                mock_chat.assert_called_once()

    def test_init_with_custom_model(self, mock_settings):
        """GeminiLLM should accept custom model name."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                llm = GeminiLLM(model_name="custom-model")

                assert llm.model_name == "custom-model"

    def test_init_with_custom_temperature(self, mock_settings):
        """GeminiLLM should accept custom temperature."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                llm = GeminiLLM(temperature=0.5)

                assert llm.temperature == 0.5

    def test_invoke(self, mock_settings):
        """invoke should return response from model."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                mock_model = MagicMock()
                mock_model.invoke.return_value = MagicMock(content="Test response")
                mock_chat.return_value = mock_model

                llm = GeminiLLM()
                response = llm.invoke("Test prompt")

                assert response == "Test response"
                mock_model.invoke.assert_called_once()

    def test_invoke_with_messages(self, mock_settings):
        """invoke should handle message list input."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                mock_model = MagicMock()
                mock_model.invoke.return_value = MagicMock(content="Response to messages")
                mock_chat.return_value = mock_model

                llm = GeminiLLM()
                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ]
                response = llm.invoke(messages)

                assert response == "Response to messages"

    @pytest.mark.asyncio
    async def test_ainvoke(self, mock_settings):
        """ainvoke should return response asynchronously."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                mock_model = MagicMock()
                mock_model.ainvoke = AsyncMock(
                    return_value=MagicMock(content="Async response")
                )
                mock_chat.return_value = mock_model

                llm = GeminiLLM()
                response = await llm.ainvoke("Test prompt")

                assert response == "Async response"

    def test_stream(self, mock_settings):
        """stream should yield response chunks."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                mock_model = MagicMock()
                mock_model.stream.return_value = iter(
                    [
                        MagicMock(content="Hello"),
                        MagicMock(content=" world"),
                        MagicMock(content="!"),
                    ]
                )
                mock_chat.return_value = mock_model

                llm = GeminiLLM()
                chunks = list(llm.stream("Test prompt"))

                assert len(chunks) == 3
                assert "".join(chunks) == "Hello world!"

    def test_get_model(self, mock_settings):
        """get_model should return the underlying LangChain model."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                mock_model = MagicMock()
                mock_chat.return_value = mock_model

                llm = GeminiLLM()
                model = llm.get_model()

                assert model is mock_model

    def test_with_callbacks(self, mock_settings):
        """GeminiLLM should support callbacks."""
        with patch("src.llm.gemini.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("src.llm.gemini.ChatGoogleGenerativeAI") as mock_chat:
                callback = MagicMock()
                llm = GeminiLLM(callbacks=[callback])

                assert llm.callbacks == [callback]
