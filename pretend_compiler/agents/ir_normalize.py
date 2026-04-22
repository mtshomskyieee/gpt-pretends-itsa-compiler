"""Intermediate representation (IR): the fictional compiler output for this tool—a JSON array of
op dicts (e.g. ``print``, ``tool``, ``emit_stderr``) that ``pretend_compiler.agents.vm_runner``
interprets. It is not machine code or source; it is the staged form between lowered source and VM
execution. This module normalizes that IR so ``pretend-linked.ll`` and VM runs stay stable across
LLM samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretend_compiler.vm_tool_registry import VM_SANDBOX_TOOL_NAMES

# Backwards-compatible alias; must match ``vm_tools.make_vm_tools``.
ALLOWED_VM_TOOL_NAMES = VM_SANDBOX_TOOL_NAMES


def _reserved_stream_path(path: str) -> bool:
    """True if path looks like stdio device names models misuse as filenames."""
    if not path or not str(path).strip():
        return False
    p = str(path).strip().replace("\\", "/").lower()
    base = Path(p).name
    if base in ("stdout", "stderr", "stdin", "-", "con", "nul"):
        return True
    if p in ("/dev/stdout", "/dev/stderr", "/dev/stdin"):
        return True
    return False


def sanitize_ir_ops(ops: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    - **print / emit_stderr**: ensure each non-empty payload ends with ``\\n`` so consecutive ops do not glue.
    - **tool write_file**: drop ops that target reserved stream-like paths (models sometimes emit
      ``write_file(path=\"stdout\")`` which produces garbage like ``ok wrote N bytes`` on stdout).
    - **tool name**: drop ``op: tool`` when ``name`` is not one of the allowed VM tools (models sometimes emit
      source builtins like ``range`` or ``print`` as tool names, which would pollute stdout with errors).
    """
    out: list[dict[str, Any]] = []
    for raw in ops or []:
        if not isinstance(raw, dict):
            continue
        op = dict(raw)
        kind = op.get("op")

        if kind == "tool":
            name = str(op.get("name", ""))
            if name not in ALLOWED_VM_TOOL_NAMES:
                continue
            args = dict(op.get("args") or {})
            path = args.get("path") or args.get("filename") or ""
            if name == "write_file" and _reserved_stream_path(str(path)):
                continue
            out.append(op)
            continue

        if kind == "print":
            text = str(op.get("text", ""))
            if text and not text.endswith("\n"):
                op["text"] = text + "\n"
            out.append(op)
            continue

        if kind == "emit_stderr":
            text = str(op.get("text", ""))
            if text and not text.endswith("\n"):
                op["text"] = text + "\n"
            out.append(op)
            continue

        out.append(op)

    return out
