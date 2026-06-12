from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    whisper_model: str = "whisper-large-v3"

    max_file_size_mb: int = 25
    request_timeout_s: float = 60.0
    llm_timeout_s: float = 60.0
    max_context_chars: int = 6000
    max_tool_input_chars: int = 16000
    max_conversation_chars: int = 12000

    port: int = 8000
    host: str = "0.0.0.0"
    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def whisper_configured(self) -> bool:
        return bool(self.groq_api_key or (self.llm_api_key and "groq" in self.llm_base_url))

    def whisper_key(self) -> str:
        return self.groq_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
