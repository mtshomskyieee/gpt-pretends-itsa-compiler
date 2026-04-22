"""Graph integration with mocked LLM nodes."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretend_compiler.graph import make_pipeline


def test_pipeline_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pretend_compiler import graph as graph_mod

    def fake_pre(llm, state):
        return {
            "validation_ok": True,
            "validation_diagnostics": [],
            "plan": "step1",
        }

    def fake_comp(llm, state):
        return {
            "compile_ok": True,
            "compiler_diagnostics": [],
            "ir_ops": [{"op": "print", "text": "mock"}],
        }

    monkeypatch.setattr(graph_mod, "run_precompile", fake_pre)
    monkeypatch.setattr(graph_mod, "run_compile", fake_comp)

    pipe = make_pipeline(MagicMock())
    p = tmp_path / "a.c"
    p.write_text("int main(){return 0;}\n")
    out = pipe.invoke(
        {
            "lang": "c",
            "source_path": str(p),
            "source_text": p.read_text(),
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": True,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )
    assert out.get("exit_code") == 0
    out_text = out.get("vm_stdout") or ""
    assert "PLAN" in out_text
    assert "step1" in out_text
    assert "ir_ops" in out_text or "print" in out_text


def test_validation_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pretend_compiler import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "run_precompile",
        lambda llm, s: {
            "validation_ok": False,
            "validation_diagnostics": [{"message": "bad", "severity": "error"}],
            "plan": "",
        },
    )

    pipe = make_pipeline(MagicMock())
    p = tmp_path / "bad.py"
    p.write_text("+++")
    out = pipe.invoke(
        {
            "lang": "python",
            "source_path": str(p),
            "source_text": p.read_text(),
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": False,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )
    assert int(out.get("exit_code") or 0) == 1
    stderr = out.get("vm_stderr") or ""
    assert "[validation]" in stderr
    assert "bad" in stderr


def test_compile_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "compile_ok": False,
            "compiler_diagnostics": [{"message": "cannot lower", "severity": "error"}],
            "ir_ops": [],
        },
    )

    pipe = make_pipeline(MagicMock())
    p = tmp_path / "a.c"
    p.write_text("int main(){return 0;}\n")
    out = pipe.invoke(
        {
            "lang": "c",
            "source_path": str(p),
            "source_text": p.read_text(),
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": False,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )
    assert int(out.get("exit_code") or 0) == 1
    stderr = out.get("vm_stderr") or ""
    assert "[compiler]" in stderr
    assert "cannot lower" in stderr


def test_vm_recoverable_fault_retries_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pretend_compiler import graph as graph_mod

    compile_calls: list[int] = []

    def fake_pre(llm, state):
        return {"validation_ok": True, "validation_diagnostics": [], "plan": "plan"}

    def fake_comp(llm, state):
        r = int(state.get("retries") or 0)
        compile_calls.append(r)
        if r == 0:
            return {
                "compile_ok": True,
                "compiler_diagnostics": [],
                "ir_ops": [
                    {
                        "op": "fault",
                        "message": "try again",
                        "recoverable": True,
                        "hint": "emit exit 0",
                    },
                ],
            }
        return {
            "compile_ok": True,
            "compiler_diagnostics": [],
            "ir_ops": [{"op": "print", "text": "fixed\n"}, {"op": "exit", "code": 0}],
        }

    monkeypatch.setattr(graph_mod, "run_precompile", fake_pre)
    monkeypatch.setattr(graph_mod, "run_compile", fake_comp)

    pipe = make_pipeline(MagicMock())
    p = tmp_path / "x.c"
    p.write_text("int main(){return 0;}\n")
    out = pipe.invoke(
        {
            "lang": "c",
            "source_path": str(p),
            "source_text": p.read_text(),
            "hosted": False,
            "backend": "mistral",
            "allow_network": False,
            "sandbox_dir": str(tmp_path),
            "max_retries": 2,
            "retries": 0,
            "dry_run": False,
            "validation_diagnostics": [],
            "compiler_diagnostics": [],
            "ir_ops": [],
        }
    )
    assert compile_calls == [0, 1]
    assert int(out.get("exit_code") or 0) == 0
    assert (out.get("vm_stdout") or "") == "fixed\n"
    assert int(out.get("retries") or 0) == 1
