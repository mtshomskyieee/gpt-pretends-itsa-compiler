"""Shared LangGraph state for the pretend compile pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class PretendState(TypedDict, total=False):
    """End-to-end state for precompiler → compiler → VM."""

    lang: str
    source_path: str
    source_text: str

    hosted: bool
    backend: str  # mistral | qwen | ollama

    validation_ok: bool
    validation_diagnostics: list[dict[str, Any]]

    plan: str

    compile_ok: bool
    compiler_diagnostics: list[dict[str, Any]]
    ir_ops: list[dict[str, Any]]

    vm_stdout: str
    vm_stderr: str
    vm_fault_recoverable: bool
    vm_fault_hint: str
    exit_code: int

    retries: int
    max_retries: int

    allow_network: bool
    sandbox_dir: str
    vm_stdin: str  # pretend process stdin for read_stdin (line buffer)
    vm_random_seed: int | None  # seeds random_int/random_float tools (deterministic runs)

    dry_run: bool
    use_files: bool
    pretend_linked_path: str
    pretend_plan_path: str

    error_message: str | None
