"""Deterministic language gate helpers (Python brace / ast)."""

import textwrap

from pretend_compiler.agents.precompiler import _deterministic_brace_balance, _deterministic_python


def test_python_ast_invalid_indent() -> None:
    src = textwrap.dedent(
        """\
        def f():
        x = 1
        """
    )
    ok, diags = _deterministic_python(src)
    assert not ok
    assert diags and diags[0].message


def test_python_valid() -> None:
    ok, diags = _deterministic_python("print(1)\n")
    assert ok
    assert not diags


def test_brace_balance_c() -> None:
    ok, diags = _deterministic_brace_balance("c", "int main() { return 0; ")
    assert not ok
    assert diags
