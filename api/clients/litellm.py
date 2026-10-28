import os
from typing import Any, Dict, Optional

from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.core.types import ModelType
from openai import AsyncOpenAI, OpenAI


class LiteLLMClient(OpenAIClient):
    """
    LiteLLM OpenAI-compatible client.

    LiteLLM exposes an OpenAI-compatible API surface, so we reuse
    OpenAIClient for embeddings. LLM calls use the Chat Completions API
    explicitly: adalflow>=1.1 switched OpenAIClient to the Responses API,
    while the chat streamers consume chat-completion chunks.

    Expected environment variables:

    LITELLM_BASE_URL=http://litellm:4000
    LITELLM_API_KEY=sk-1234

    Example model names:
        openai/gpt-4o
        anthropic/claude-3-5-sonnet
        gemini/gemini-2.5-pro
        ollama/llama3
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        input_type: str = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "LITELLM_BASE_URL",
        env_api_key_name: str = "LITELLM_API_KEY",
    ):
        resolved_base_url = base_url or os.getenv(
            env_base_url_name, "http://localhost:4000"
        )
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url.rstrip('/')}/v1"
        # Only pass arguments accepted by every supported adalflow version.
        super().__init__(
            api_key=api_key,
            input_type=input_type,
            base_url=resolved_base_url,
            env_api_key_name=env_api_key_name,
        )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        if model_type != ModelType.LLM:
            return super().convert_inputs_to_api_kwargs(input, model_kwargs, model_type)
        messages = (
            input if isinstance(input, list) else [{"role": "user", "content": input}]
        )
        return {**model_kwargs, "messages": messages}

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if model_type != ModelType.LLM:
            return super().call(api_kwargs, model_type)
        self._api_kwargs = api_kwargs
        return self.sync_client.chat.completions.create(**api_kwargs)

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if model_type != ModelType.LLM:
            return await super().acall(api_kwargs, model_type)
        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()
        return await self.async_client.chat.completions.create(**api_kwargs)

    def init_sync_client(self):
        """
        Initialize synchronous LiteLLM OpenAI-compatible client.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name, "dummy")
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def init_async_client(self):
        """
        Initialize asynchronous LiteLLM OpenAI-compatible client.
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name, "dummy")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
