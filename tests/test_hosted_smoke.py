"""Optional hosted-API smoke tests (set ``OPENAI_API_KEY`` in CI or locally)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.hosted


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_hosted_dry_run_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "hello.py"
    cmd = [
        sys.executable,
        "-m",
        "pretend_compiler",
        "--hosted",
        "--lang",
        "python",
        str(fixture),
        "--dry-run",
    ]
    env = dict(os.environ)
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    combined = proc.stdout + proc.stderr
    assert "PLAN" in combined or "IR" in combined
