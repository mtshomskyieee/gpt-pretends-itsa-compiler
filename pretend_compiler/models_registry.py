"""Pinned Hugging Face GGUF artifacts for `pretend-models pull`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGUFArtifact:
    repo_id: str
    filename: str
    description: str
    # Additional blobs in the same repo/dir (e.g. shard 2 of 2); loader uses ``filename``.
    extra_filenames: tuple[str, ...] = ()


# Apache-2.0 Mistral 7B Instruct — community GGUF build (widely used).
MISTRAL_DEFAULT = GGUFArtifact(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    description="Mistral 7B Instruct v0.3 Q4_K_M",
)

# Qwen2.5 — use official GGUF repo (Apache-2.0). Q4_K_M is split into two parts on HF.
QWEN_DEFAULT = GGUFArtifact(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    description="Qwen2.5 7B Instruct Q4_K_M",
    extra_filenames=("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",),
)


def resolve_artifact(name: str) -> GGUFArtifact:
    key = name.strip().lower().replace("-", "_")
    if key in ("mistral", "mistral_default", "default"):
        return MISTRAL_DEFAULT
    if key in ("qwen", "qwen_default", "qwen2.5"):
        return QWEN_DEFAULT
    raise ValueError(f"Unknown model alias: {name!r}. Use 'mistral' or 'qwen'.")
