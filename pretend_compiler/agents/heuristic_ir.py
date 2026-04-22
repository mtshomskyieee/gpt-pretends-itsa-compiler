"""Deterministic IR when the LLM returns empty ops — helps CPU-only / small GGUF runs."""

from __future__ import annotations

import re


def _c_unescape_literal(s: str) -> str:
    """Minimal C string literal unescape for common escapes."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\\\', "\\")
        .replace('\\"', '"')
    )


def heuristic_c_scanf_two_int_divide(source: str) -> list[dict] | None:
    """
    Deterministic lowering for a tiny C template: ``scanf("%d %d")``, divide ``(double)a/(double)b``,
    ``printf(...%f...)``. Hosted models often emit *linear* IR that includes both failure branches and
    multiple ``exit`` ops; this IR is a single straight-line path suitable for the pretend VM.
    """
    if "scanf" not in source or "printf" not in source:
        return None
    if not re.search(r"scanf\s*\(\s*\"[^\"]*%d[^\"]*%d", source):
        return None
    if not re.search(r"\(double\)\s*a\s*/\s*\(double\)\s*b", source):
        return None

    result_lit: str | None = None
    for m in re.finditer(r'printf\s*\(\s*"((?:[^"\\]|\\.)*)"', source):
        raw = m.group(1)
        if re.search(r"%[fgGeE]", raw):
            result_lit = _c_unescape_literal(raw)
    if not result_lit:
        return None

    idx_scanf = source.find("scanf")
    if idx_scanf < 0:
        return None
    head = source[:idx_scanf]
    prompt_lit: str | None = None
    for m in re.finditer(r'printf\s*\(\s*"((?:[^"\\]|\\.)*)"', head):
        prompt_lit = _c_unescape_literal(m.group(1))
    if prompt_lit is None:
        prompt_lit = "Enter two integers: "
    if not prompt_lit.endswith("\n"):
        prompt_lit = prompt_lit + "\n"

    out_fmt = result_lit if result_lit.endswith("\n") else result_lit + "\n"
    return [
        {"op": "print", "text": prompt_lit},
        {"op": "flush_stdout"},
        {"op": "tool", "name": "read_stdin", "args": {}, "to_stdout": False},
        {"op": "print", "text": out_fmt},
        {"op": "exit", "code": 0},
    ]


def _heuristic_c_printf_loop(source: str) -> list[dict] | None:
    """
    Match ``for (..; .. < N; ..) { ... printf("..."); ... }`` style loops.

    Returns IR or None if pattern does not match.
    """
    # Upper bound in for (init; cond; step) — e.g. i < 10
    bound_m = re.search(
        r"for\s*\([^;]*;\s*[^;]*?<\s*(\d+)\s*;",
        source,
        re.MULTILINE | re.DOTALL,
    )
    if not bound_m:
        return None
    n = int(bound_m.group(1))
    if n < 1 or n > 10_000:
        return None

    # First printf string literal in the loop body region (best-effort)
    printf_m = re.search(r'printf\s*\(\s*"((?:[^"\\]|\\.)*)"\s*\)', source, re.DOTALL)
    if not printf_m:
        return None
    text = _c_unescape_literal(printf_m.group(1))

    ops: list[dict] = [{"op": "print", "text": text} for _ in range(n)]
    ops.append({"op": "exit", "code": 0})
    return ops


def _heuristic_python_print_loop(source: str) -> list[dict] | None:
    """Match ``for _ in range(N):`` + ``print("...")`` / ``print('...')``."""
    rm = re.search(
        r"for\s+\w+\s+in\s+range\s*\(\s*(\d+)\s*\)\s*:",
        source,
    )
    if not rm:
        return None
    n = int(rm.group(1))
    if n < 1 or n > 10_000:
        return None

    pm = re.search(r"print\s*\(\s*(['\"])(.*?)\1\s*\)", source, re.DOTALL)
    if not pm:
        return None
    text = pm.group(2).replace("\\n", "\n")
    # Match default ``print``: one line of output per iteration (newline appended).
    line = text if text.endswith("\n") else text + "\n"

    ops: list[dict] = [{"op": "print", "text": line} for _ in range(n)]
    ops.append({"op": "exit", "code": 0})
    return ops


def maybe_fix_under_unrolled_loop(
    lang: str, source: str, ir_ops: list[dict]
) -> tuple[list[dict], bool]:
    """
    When the source matches deterministic loop lowering but the model returned too few ``print`` ops
    (common with small GGUF models), substitute full heuristic IR.
    """
    heur = synthesize_stdout_ir(lang, source)
    if not heur:
        return ir_ops, False
    expected_prints = sum(1 for op in heur if op.get("op") == "print")
    actual_prints = sum(1 for op in ir_ops if op.get("op") == "print")
    if actual_prints >= expected_prints:
        return ir_ops, False
    return heur, True


def synthesize_stdout_ir(lang: str, source: str) -> list[dict] | None:
    """If source matches a trivial stdout pattern, return IR; else None."""
    lang_l = (lang or "").strip().lower()
    if lang_l in {"c"}:
        div = heuristic_c_scanf_two_int_divide(source)
        if div:
            return div
        return _heuristic_c_printf_loop(source)
    if lang_l in {"python", "py"}:
        return _heuristic_python_print_loop(source)
    # C++ often uses same printf loop — try C heuristic
    if lang_l in {"cpp", "c++", "cxx"}:
        return _heuristic_c_printf_loop(source)
    return None
