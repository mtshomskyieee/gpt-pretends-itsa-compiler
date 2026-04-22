"""--use-files: IR written to pretend-linked.ll and VM reads from disk."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretend_compiler.graph import (
    PRETEND_LINK_BASENAME,
    PRETEND_PLAN_BASENAME,
    make_pipeline,
)


def test_use_files_vm_reads_linked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pretend_compiler import graph as graph_mod

    def fake_pre(llm, state):
        return {"validation_ok": True, "validation_diagnostics": [], "plan": "p"}

    def fake_comp(llm, state):
        return {
            "compile_ok": True,
            "compiler_diagnostics": [],
            "ir_ops": [
                {"op": "print", "text": "x"},
                {"op": "exit", "code": 0},
            ],
        }

    monkeypatch.setattr(graph_mod, "run_precompile", fake_pre)
    monkeypatch.setattr(graph_mod, "run_compile", fake_comp)

    pipe = make_pipeline(MagicMock())
    src = tmp_path / "a.c"
    src.write_text("int main(){return 0;}\n")
    out = pipe.invoke(
        {
            "lang": "c",
            "source_path": str(src),
            "source_text": src.read_text(),
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": False,
            "use_files": True,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )

    plan_file = tmp_path / PRETEND_PLAN_BASENAME
    assert plan_file.is_file()
    assert plan_file.read_text(encoding="utf-8") == "p"

    link = tmp_path / PRETEND_LINK_BASENAME
    assert link.is_file()
    data = json.loads(link.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert out.get("exit_code") == 0
    assert (out.get("vm_stdout") or "") == "x\n"


def test_dry_run_use_files_writes_linked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pretend_compiler import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "run_precompile",
        lambda llm, s: {"validation_ok": True, "validation_diagnostics": [], "plan": "p"},
    )
    monkeypatch.setattr(
        graph_mod,
        "run_compile",
        lambda llm, s: {
            "compile_ok": True,
            "compiler_diagnostics": [],
            "ir_ops": [{"op": "print", "text": "hi"}],
        },
    )

    pipe = make_pipeline(MagicMock())
    src = tmp_path / "x.c"
    src.write_text("x")
    out = pipe.invoke(
        {
            "lang": "c",
            "source_path": str(src),
            "source_text": "x",
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": True,
            "use_files": True,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )
    assert out.get("exit_code") == 0
    assert (tmp_path / PRETEND_PLAN_BASENAME).is_file()
    assert (tmp_path / PRETEND_LINK_BASENAME).is_file()
    assert "Wrote IR" in (out.get("vm_stdout") or "")
