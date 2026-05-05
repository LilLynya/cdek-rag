from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения. Значения берутся из переменных окружения и/или .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    llm_provider: str = Field(
        default="openai",
        description="Провайдер LLM: openai, ollama, anthropic, groq, together, ...",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Имя модели у выбранного провайдера.",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ),
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Базовый URL для OpenAI-совместимых эндпоинтов "
            "(LM Studio, vLLM, Ollama OpenAI-mode, Together, и т.п.)."
        ),
    )
    llm_temperature: float = 0.0

    # ---- Knowledge base ----
    data_dir: Path = Path("data")

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
