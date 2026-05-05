"""Фабрика LLM. Подключает любую модель — локальную или облачную — через единый интерфейс."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import settings


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """Возвращает chat-модель LangChain согласно настройкам.

    Поддержка провайдеров:
      * openai     — `langchain-openai` (включая совместимые эндпоинты через `LLM_BASE_URL`).
      * ollama     — `langchain-ollama` (опциональная зависимость).
      * anthropic  — `langchain-anthropic` (опциональная зависимость).
      * прочее     — fallback в `langchain.chat_models.init_chat_model`.
    """
    provider = settings.llm_provider.lower().strip()
    temp = settings.llm_temperature if temperature is None else temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Для провайдера 'ollama' установите пакет langchain-ollama"
            ) from exc

        return ChatOllama(
            model=settings.llm_model,
            temperature=temp,
            base_url=settings.llm_base_url or "http://localhost:11434",
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Для провайдера 'anthropic' установите пакет langchain-anthropic"
            ) from exc

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.llm_api_key,
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=settings.llm_model,
        model_provider=provider,
        temperature=temp,
    )
