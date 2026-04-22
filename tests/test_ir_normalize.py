"""IR sanitization for stable pretend-linked.ll and stdout."""

from pretend_compiler.agents.ir_normalize import sanitize_ir_ops


def test_print_adds_newline() -> None:
    ops = [{"op": "print", "text": "hello world"}, {"op": "exit", "code": 0}]
    out = sanitize_ir_ops(ops)
    assert out[0]["text"].endswith("\n")


def test_drop_invalid_tool_name() -> None:
    ops = [
        {"op": "tool", "name": "range", "args": {"n": 10}, "to_stdout": False},
        {"op": "print", "text": "x\n"},
    ]
    out = sanitize_ir_ops(ops)
    assert len(out) == 1
    assert out[0]["op"] == "print"


def test_drop_write_file_to_stdout() -> None:
    ops = [
        {
            "op": "tool",
            "name": "write_file",
            "args": {"path": "stdout", "content": "x"},
        },
        {"op": "print", "text": "ok\n"},
    ]
    out = sanitize_ir_ops(ops)
    assert len(out) == 1
    assert out[0]["op"] == "print"


def test_preserves_real_write_file() -> None:
    ops = [
        {
            "op": "tool",
            "name": "write_file",
            "args": {"path": "out.txt", "content": "x"},
        },
    ]
    assert len(sanitize_ir_ops(ops)) == 1
