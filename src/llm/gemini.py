"""Gemini LLM integration."""

from typing import Any, Iterator, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings


class GeminiLLM:
    """Wrapper for Gemini LLM via LangChain."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
    ):
        """Initialize Gemini LLM.

        Args:
            model_name: The model name. Defaults to settings.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            callbacks: Optional callbacks for tracing.
        """
        settings = get_settings()
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.callbacks = callbacks or []

        self._model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.google_api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            callbacks=self.callbacks,
        )

    def invoke(
        self,
        input_data: Union[str, list],
    ) -> str:
        """Invoke the model with input.

        Args:
            input_data: Either a string prompt or list of messages.

        Returns:
            The model's response text.
        """
        messages = self._prepare_messages(input_data)
        response = self._model.invoke(messages)
        return response.content

    async def ainvoke(
        self,
        input_data: Union[str, list],
    ) -> str:
        """Invoke the model asynchronously.

        Args:
            input_data: Either a string prompt or list of messages.

        Returns:
            The model's response text.
        """
        messages = self._prepare_messages(input_data)
        response = await self._model.ainvoke(messages)
        return response.content

    def stream(
        self,
        input_data: Union[str, list],
    ) -> Iterator[str]:
        """Stream responses from the model.

        Args:
            input_data: Either a string prompt or list of messages.

        Yields:
            Response text chunks.
        """
        messages = self._prepare_messages(input_data)
        for chunk in self._model.stream(messages):
            yield chunk.content

    async def astream(
        self,
        input_data: Union[str, list],
    ):
        """Stream responses asynchronously.

        Args:
            input_data: Either a string prompt or list of messages.

        Yields:
            Response text chunks.
        """
        messages = self._prepare_messages(input_data)
        async for chunk in self._model.astream(messages):
            yield chunk.content

    def get_model(self) -> ChatGoogleGenerativeAI:
        """Get the underlying LangChain model.

        Returns:
            The ChatGoogleGenerativeAI instance.
        """
        return self._model

    def _prepare_messages(
        self,
        input_data: Union[str, list],
    ) -> list:
        """Prepare messages for the model.

        Args:
            input_data: A string, list of message dicts, or list of BaseMessage objects.

        Returns:
            List of LangChain message objects.
        """
        if isinstance(input_data, str):
            return [HumanMessage(content=input_data)]

        if input_data and isinstance(input_data[0], BaseMessage):
            return list(input_data)

        messages = []
        for msg in input_data:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        return messages
