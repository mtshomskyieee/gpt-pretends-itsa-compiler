"""Single source of truth for VM sandbox tool names (IR ``op: tool`` / ``vm_tools``)."""

from __future__ import annotations

# Keep in sync with ``make_vm_tools`` registrations in ``pretend_compiler.agents.vm_tools``.
VM_SANDBOX_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "read_bytes",
        "write_bytes",
        "getenv",
        "http_get",
        "http_post",
        "read_stdin",
        "list_dir",
        "path_exists",
        "remove_file",
        "rename",
        "mkdir",
        "sleep_ms",
        "time_now",
        "random_int",
        "random_float",
    }
)


def format_vm_tool_names_for_compiler_prompt() -> str:
    """Comma-separated backtick names for the compiler system prompt (one line)."""
    return ", ".join(f"`{n}`" for n in sorted(VM_SANDBOX_TOOL_NAMES))


def format_vm_tool_names_for_fallback_schema() -> str:
    """``"a"|"b"|…`` fragment for the JSON-fallback prompt (pseudo-schema)."""
    return "|".join(f'"{n}"' for n in sorted(VM_SANDBOX_TOOL_NAMES))
