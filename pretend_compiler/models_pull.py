"""Download pinned GGUF weights or shell out to `ollama pull`."""

from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from pretend_compiler.llm_factory import models_cache_root
from pretend_compiler.models_registry import (
    GGUFArtifact,
    MISTRAL_DEFAULT,
    QWEN_DEFAULT,
    resolve_artifact,
)


def env_key_for_artifact(art: GGUFArtifact) -> str:
    if art is MISTRAL_DEFAULT:
        return "MISTRAL_GGUF_PATH"
    if art is QWEN_DEFAULT:
        return "QWEN_GGUF_PATH"
    return "MISTRAL_GGUF_PATH"


def llm_env_path(cwd: Path | None = None) -> Path:
    """Path to `llm.env` relative to the current working directory (default: process cwd)."""
    return (cwd or Path.cwd()).resolve() / "llm.env"


def _parse_export_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key, _, _ = stripped.partition("=")
    return key.strip()


def upsert_llm_env_gguf(env_file: Path, key: str, value: str) -> None:
    """Create or update ``llm.env``, replacing any existing assignment for ``key``."""
    lines_out: list[str] = []
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            k = _parse_export_key(line)
            if k == key:
                continue
            lines_out.append(line)
    while lines_out and lines_out[-1] == "":
        lines_out.pop()
    if lines_out:
        lines_out.append("")
    lines_out.append(f"{key}={value}")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def pull_gguf(alias: str, *, cwd: Path | None = None) -> tuple[Path, str, Path]:
    """Download GGUF, write ``MISTRAL_GGUF_PATH`` / ``QWEN_GGUF_PATH`` to ``llm.env``.

    Returns ``(resolved_gguf_path, env_key, llm_env_file)``.
    """
    art = resolve_artifact(alias)
    dest = models_cache_root() / art.repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    paths = (art.filename, *art.extra_filenames)
    resolved = Path(
        hf_hub_download(
            repo_id=art.repo_id,
            filename=paths[0],
            local_dir=str(dest),
        )
    ).resolve()
    for fn in paths[1:]:
        hf_hub_download(repo_id=art.repo_id, filename=fn, local_dir=str(dest))
    key = env_key_for_artifact(art)
    env_file = llm_env_path(cwd)
    upsert_llm_env_gguf(env_file, key, str(resolved))
    return resolved, key, env_file


def ollama_pull(model: str) -> None:
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )
    except FileNotFoundError as e:
        raise SystemExit("ollama not found on PATH; install from https://ollama.com") from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ollama pull failed: {e}") from e
