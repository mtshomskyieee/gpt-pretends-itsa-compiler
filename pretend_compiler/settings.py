from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="llm.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mistral_gguf_path: Path | None = Field(default=None, validation_alias="MISTRAL_GGUF_PATH")
    qwen_gguf_path: Path | None = Field(default=None, validation_alias="QWEN_GGUF_PATH")

    # Llama.cpp context length; models like Mistral 7B train at 32k — matching avoids n_ctx warnings.
    # Lower (e.g. 8192) via N_CTX if you hit memory limits.
    n_ctx: int = Field(default=32768, validation_alias="N_CTX")
    n_gpu_layers: int = Field(default=0, validation_alias="N_GPU_LAYERS")

    # Generation cap for local GGUF (default ChatLlamaCpp max_tokens is only 256 — too small for IR JSON).
    llm_max_tokens: int = Field(default=8192, validation_alias="LLM_MAX_TOKENS")

    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="mistral", validation_alias="OLLAMA_MODEL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")


def load_settings() -> Settings:
    return Settings()
