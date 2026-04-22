"""Tests for pretend stdin resolution (TTY vs pipe)."""

import io
import os
import stat
import sys
from pathlib import Path

import pytest

from pretend_compiler.cli import _read_stdin_for_dash, _resolve_vm_stdin


def test_stdin_text_wins() -> None:
    assert _resolve_vm_stdin(stdin_text="x", stdin_file=Path("-")) == "x"


def test_stdin_dash_tty_reads_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_in = io.StringIO("9 2\nrest")
    fake_in.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", fake_in)
    assert _resolve_vm_stdin(stdin_text="", stdin_file=Path("-")) == "9 2\n"


def test_stdin_dash_pipe_reads_all(monkeypatch: pytest.MonkeyPatch) -> None:
    read_end, write_end = os.pipe()
    os.write(write_end, b"a\nb\n")
    os.close(write_end)
    pipe_in = open(read_end, "r", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", pipe_in)
    try:
        assert stat.S_ISFIFO(os.fstat(pipe_in.fileno()).st_mode)
        assert _resolve_vm_stdin(stdin_text="", stdin_file=Path("-")) == "a\nb\n"
    finally:
        pipe_in.close()


def test_stdin_dash_non_tty_non_fifo_reads_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDE / subprocess without isatty or FIFO must not use blocking read() until EOF."""
    fake_in = io.StringIO("only\nmore")

    def fake_isatty() -> bool:
        return False

    fake_in.isatty = fake_isatty  # type: ignore[method-assign]

    def boom(_fd: int) -> None:
        raise OSError("no fileno")

    monkeypatch.setattr(sys, "stdin", fake_in)
    monkeypatch.setattr("pretend_compiler.cli.os.fstat", boom)
    assert _read_stdin_for_dash() == "only\n"
