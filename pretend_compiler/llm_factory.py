from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from pretend_compiler.models_registry import MISTRAL_DEFAULT, QWEN_DEFAULT
from pretend_compiler.settings import Settings

Backend = Literal["mistral", "qwen", "ollama"]


def models_cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        root = Path(xdg) / "pretend_compiler" / "models"
    else:
        root = Path.home() / ".cache" / "pretend_compiler" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_gguf_path(settings: Settings, backend: Backend) -> Path:
    artifact = MISTRAL_DEFAULT if backend == "mistral" else QWEN_DEFAULT
    explicit = settings.mistral_gguf_path if backend == "mistral" else settings.qwen_gguf_path
    if explicit and Path(explicit).is_file():
        return Path(explicit).resolve()

    searched: list[str] = []
    for base in (
        models_cache_root(),
        models_cache_root() / artifact.repo_id.replace("/", "__"),
    ):
        p = base / artifact.filename
        searched.append(str(p))
        if p.is_file():
            return p.resolve()

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.is_dir():
        try:
            for gguf in hf_cache.rglob(artifact.filename):
                if gguf.is_file():
                    return gguf.resolve()
        except OSError:
            pass

    msg = (
        f"GGUF not found for backend={backend!r} ({artifact.filename}). "
        f"Searched: {searched}. Run: pretend-models pull {backend}"
    )
    raise FileNotFoundError(msg)


def build_chat_model(
    *,
    settings: Settings,
    hosted: bool,
    backend: Backend,
) -> BaseChatModel:
    """When ``hosted`` is True, ``backend`` is ignored for provider selection."""
    if hosted:
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for --hosted (set in llm.env or environment)."
            )
        kwargs: dict = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "Install Ollama extra: uv sync --extra ollama"
            ) from e
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    from langchain_community.chat_models import ChatLlamaCpp

    path = resolve_gguf_path(settings, backend)
    return ChatLlamaCpp(
        model_path=str(path),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        max_tokens=settings.llm_max_tokens,
        verbose=False,
    )
