from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, settings
from app.core.exceptions import LLMProviderError


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the model's answer text, or raises LLMProviderError on failure."""


class OpenAIChatProvider(LLMProvider):
    def __init__(self, cfg: Settings):
        if not cfg.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for llm_provider=openai")
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.openai_chat_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"OpenAI chat request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("OpenAI returned an empty response")
        return content


class AnthropicChatProvider(LLMProvider):
    def __init__(self, cfg: Settings):
        if not cfg.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is required for llm_provider=anthropic")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.anthropic_chat_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Anthropic chat request failed: {exc}") from exc

        if not response.content:
            raise LLMProviderError("Anthropic returned an empty response")
        return response.content[0].text


@lru_cache
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "anthropic":
        return AnthropicChatProvider(settings)
    return OpenAIChatProvider(settings)
