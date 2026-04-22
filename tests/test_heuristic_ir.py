"""Deterministic heuristic IR for trivial loops."""

from pathlib import Path

from pretend_compiler.agents.heuristic_ir import (
    heuristic_c_scanf_two_int_divide,
    maybe_fix_under_unrolled_loop,
    synthesize_stdout_ir,
)


def test_sample_c_heuristic() -> None:
    src = Path(__file__).resolve().parents[1] / "sample.c"
    text = src.read_text(encoding="utf-8")
    ops = synthesize_stdout_ir("c", text)
    assert ops is not None
    assert len(ops) == 11  # 10 print + exit
    assert ops[-1] == {"op": "exit", "code": 0}
    prints = [o for o in ops if o.get("op") == "print"]
    assert len(prints) == 10
    assert all("hello world" in p["text"] for p in prints)


def test_sample_divide_heuristic() -> None:
    src = Path(__file__).resolve().parents[1] / "sample_divide.c"
    text = src.read_text(encoding="utf-8")
    ops = heuristic_c_scanf_two_int_divide(text)
    assert ops is not None
    assert ops[-1] == {"op": "exit", "code": 0}
    kinds = [o["op"] for o in ops]
    assert kinds == ["print", "flush_stdout", "tool", "print", "exit"]
    assert ops[2]["name"] == "read_stdin"
    assert "%f" in ops[3]["text"]
    ops2 = synthesize_stdout_ir("c", text)
    assert ops2 == ops


def test_sample_py_heuristic() -> None:
    src = Path(__file__).resolve().parents[1] / "sample.py"
    text = src.read_text(encoding="utf-8")
    ops = synthesize_stdout_ir("python", text)
    assert ops is not None
    assert len(ops) == 11


def test_maybe_fix_under_unrolled_loop_replaces_sparse_model_ir() -> None:
    """GGUF models sometimes emit one print for a counted loop — patch to full unroll."""
    src = Path(__file__).resolve().parents[1] / "sample.c"
    text = src.read_text(encoding="utf-8")
    bad_ir = [{"op": "print", "text": "hello world\n"}]
    fixed, used = maybe_fix_under_unrolled_loop("c", text, bad_ir)
    assert used is True
    assert sum(1 for op in fixed if op.get("op") == "print") == 10


def test_maybe_fix_noop_when_already_full() -> None:
    src = Path(__file__).resolve().parents[1] / "sample.c"
    text = src.read_text(encoding="utf-8")
    full = synthesize_stdout_ir("c", text)
    assert full is not None
    same, used = maybe_fix_under_unrolled_loop("c", text, full)
    assert used is False
    assert same == full
