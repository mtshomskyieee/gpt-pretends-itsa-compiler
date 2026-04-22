"""Tests for automatic llm.env updates after pretend-models pull."""

from pathlib import Path

import pytest

from pretend_compiler.models_pull import llm_env_path, upsert_llm_env_gguf


def test_upsert_creates_and_replaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = llm_env_path()
    upsert_llm_env_gguf(env, "MISTRAL_GGUF_PATH", "/first.gguf")
    text = env.read_text(encoding="utf-8")
    assert "MISTRAL_GGUF_PATH=/first.gguf" in text
    upsert_llm_env_gguf(env, "MISTRAL_GGUF_PATH", "/second.gguf")
    text = env.read_text(encoding="utf-8")
    assert text.count("MISTRAL_GGUF_PATH") == 1
    assert "/second.gguf" in text


def test_upsert_preserves_other_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = llm_env_path()
    env.write_text("N_CTX=4096\n", encoding="utf-8")
    upsert_llm_env_gguf(env, "MISTRAL_GGUF_PATH", "/m.gguf")
    text = env.read_text(encoding="utf-8")
    assert "N_CTX=4096" in text
    assert "MISTRAL_GGUF_PATH=/m.gguf" in text
