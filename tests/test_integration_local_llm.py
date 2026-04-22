"""Requires a downloaded GGUF or mocks; skips when no local model."""

from pathlib import Path

import pytest

from pretend_compiler.llm_factory import resolve_gguf_path
from pretend_compiler.settings import load_settings


@pytest.fixture(scope="session")
def mistral_available() -> bool:
    try:
        resolve_gguf_path(load_settings(), "mistral")
        return True
    except FileNotFoundError:
        return False


@pytest.mark.local_llm
@pytest.mark.slow
@pytest.mark.parametrize(
    "lang,file_name",
    [
        ("python", "hello.py"),
        ("cpp", "hello.cpp"),
        ("rust", "hello.rs"),
        ("c", "hello.c"),
    ],
)
def test_cli_smoke_with_gguf(
    mistral_available: bool,
    lang: str,
    file_name: str,
) -> None:
    if not mistral_available:
        pytest.skip("No Mistral GGUF present; run: pretend-models pull mistral")

    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / file_name
    cmd = [
        sys.executable,
        "-m",
        "pretend_compiler",
        "--lang",
        lang,
        str(fixture),
        "--dry-run",
    ]
    env = dict(**__import__("os").environ)
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "PLAN" in proc.stdout or "IR" in proc.stdout
