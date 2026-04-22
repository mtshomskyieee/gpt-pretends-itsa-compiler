"""Deterministic VM and tool sandbox tests."""

import base64
import json
from pathlib import Path

import pytest

from pretend_compiler.agents.vm_runner import execute_ir
from pretend_compiler.agents.vm_tools import make_vm_tools, run_tool_by_name
from pretend_compiler.vm_tool_registry import VM_SANDBOX_TOOL_NAMES


def test_make_vm_tools_matches_registry(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    assert {t.name for t in tools} == VM_SANDBOX_TOOL_NAMES


def test_getenv_empty_args_does_not_raise(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False, stdin_text="")
    out = run_tool_by_name(tools, "getenv", {})
    assert isinstance(out, str)
    assert "validation error" not in out.lower()


def test_printf_float_stub_substantiation(tmp_path: Path) -> None:
    """LLMs often lower printf(\"..%f..\", a/b) as a literal %f; VM fills from read_stdin."""
    stdout, _, code, _, _ = execute_ir(
        [
            {"op": "print", "text": "Enter: "},
            {"op": "flush_stdout"},
            {"op": "tool", "name": "read_stdin", "args": {}, "to_stdout": False},
            {"op": "print", "text": "quotient = %f\n"},
            {"op": "exit", "code": 0},
        ],
        sandbox_dir=tmp_path,
        allow_network=False,
        vm_stdin="9 2\n",
    )
    assert "4.500000" in stdout
    assert "%f" not in stdout
    assert code == 0


def test_flush_stdout_noop(tmp_path: Path) -> None:
    stdout, _, code, _, _ = execute_ir(
        [
            {"op": "print", "text": "Enter: "},
            {"op": "flush_stdout"},
            {
                "op": "tool",
                "name": "read_stdin",
                "args": {},
                "to_stdout": False,
            },
            {"op": "print", "text": "ok\n"},
            {"op": "exit", "code": 0},
        ],
        sandbox_dir=tmp_path,
        allow_network=False,
        vm_stdin="42\n",
    )
    assert stdout == "Enter: \n42\nok\n"
    assert code == 0


def test_read_stdin_injected(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False, stdin_text="9 2\n")
    a = run_tool_by_name(tools, "read_stdin", {})
    b = run_tool_by_name(tools, "read_stdin", {})
    assert a == "9 2\n"
    assert b == ""


def test_execute_ir_read_stdin(tmp_path: Path) -> None:
    """Simulate one scanf line: read_stdin then print result of division."""
    stdout, stderr, code, rec, hint = execute_ir(
        [
            {"op": "tool", "name": "read_stdin", "args": {}, "to_stdout": False},
            {"op": "print", "text": "4.5\n"},
            {"op": "exit", "code": 0},
        ],
        sandbox_dir=tmp_path,
        allow_network=False,
        vm_stdin="9 2\n",
    )
    assert "9 2" in stdout or "9 2\n" in stdout
    assert "4.5" in stdout
    assert code == 0


def test_print_and_exit(tmp_path: Path) -> None:
    stdout, stderr, code, rec, hint = execute_ir(
        [{"op": "print", "text": "hi"}, {"op": "exit", "code": 0}],
        sandbox_dir=tmp_path,
        allow_network=False,
    )
    assert stdout == "hi\n"
    assert stderr == ""
    assert code == 0
    assert not rec
    assert not hint


def test_read_write_sandbox(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    run_tool_by_name(tools, "write_file", {"path": "a.txt", "content": "abc"})
    out = run_tool_by_name(tools, "read_file", {"path": "a.txt"})
    assert "abc" in out


def test_sandbox_escape(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    out = run_tool_by_name(tools, "read_file", {"path": "../../etc/passwd"})
    assert "escapes sandbox" in out.lower() or "tool error" in out.lower()


def test_http_disabled_by_default(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    out = run_tool_by_name(tools, "http_get", {"url": "https://example.com"})
    assert "network disabled" in out.lower()


@pytest.mark.skipif(True, reason="network test — enable manually")
def test_http_enabled(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=True)
    out = run_tool_by_name(tools, "http_get", {"url": "https://example.com"})
    assert len(out) > 10


def test_printf_int_and_string_from_stdin(tmp_path: Path) -> None:
    stdout, _, code, _, _ = execute_ir(
        [
            {"op": "tool", "name": "read_stdin", "args": {}, "to_stdout": False},
            {"op": "print", "text": "n=%d s=%s\n"},
            {"op": "exit", "code": 0},
        ],
        sandbox_dir=tmp_path,
        allow_network=False,
        vm_stdin="42 hello world\n",
    )
    assert "n=42" in stdout
    assert "s=hello" in stdout
    assert code == 0


def test_fs_tools_rename_remove_list(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    run_tool_by_name(tools, "write_file", {"path": "x.txt", "content": "ok"})
    assert run_tool_by_name(tools, "path_exists", {"path": "x.txt"}) == "yes"
    names = run_tool_by_name(tools, "list_dir", {"path": "."})
    assert "x.txt" in names.splitlines()
    assert "ok renamed" in run_tool_by_name(
        tools, "rename", {"from_path": "x.txt", "to_path": "y.txt"}
    ).lower()
    assert run_tool_by_name(tools, "path_exists", {"path": "y.txt"}) == "yes"
    assert "ok removed" in run_tool_by_name(tools, "remove_file", {"path": "y.txt"}).lower()


def test_read_write_bytes_roundtrip(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    raw = b"\x00\xff\xfe"
    b64 = base64.standard_b64encode(raw).decode("ascii")
    assert "ok wrote" in run_tool_by_name(
        tools, "write_bytes", {"path": "b.bin", "content_base64": b64}
    ).lower()
    out = run_tool_by_name(tools, "read_bytes", {"path": "b.bin"})
    assert base64.standard_b64decode(out) == raw


def test_random_deterministic_with_seed(tmp_path: Path) -> None:
    t_a = make_vm_tools(tmp_path, allow_network=False, random_seed=12345)
    t_b = make_vm_tools(tmp_path, allow_network=False, random_seed=12345)
    x = run_tool_by_name(t_a, "random_int", {"low": 0, "high": 1_000_000})
    y = run_tool_by_name(t_b, "random_int", {"low": 0, "high": 1_000_000})
    assert x == y


def test_time_now_json(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    raw = run_tool_by_name(tools, "time_now", {})
    data = json.loads(raw)
    assert "epoch_ms" in data and "iso_utc" in data


def test_sleep_ms_bounded(tmp_path: Path) -> None:
    tools = make_vm_tools(tmp_path, allow_network=False)
    out = run_tool_by_name(tools, "sleep_ms", {"ms": 1})
    assert "ok slept 1" in out
