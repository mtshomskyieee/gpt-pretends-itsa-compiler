"""LangGraph wiring: precompile → compile → VM (with dry-run and optional retry)."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from pretend_compiler.agents.compiler import run_compile
from pretend_compiler.agents.ir_normalize import sanitize_ir_ops
from pretend_compiler.agents.precompiler import run_precompile
from pretend_compiler.agents.vm_runner import execute_ir
from pretend_compiler.state import PretendState

# File artifacts under ``sandbox_dir`` when using ``--use-files``.
PRETEND_LINK_BASENAME = "pretend-linked.ll"
PRETEND_PLAN_BASENAME = "pretend-linked.plan"


def _abort(state: PretendState) -> dict:
    lines: list[str] = []
    for group, label in (
        ("validation_diagnostics", "validation"),
        ("compiler_diagnostics", "compiler"),
    ):
        for d in state.get(group) or []:
            msg = d.get("message") if isinstance(d, dict) else str(d)
            if msg:
                lines.append(f"[{label}] {msg}")
    msg = "\n".join(lines) if lines else state.get("error_message") or "Compilation aborted."
    return {"vm_stderr": msg, "exit_code": 1}


def _dry_run(state: PretendState) -> dict:
    ir = sanitize_ir_ops(state.get("ir_ops") or [])
    body = json.dumps(ir, indent=2)
    extra = ""
    if state.get("use_files"):
        sandbox = Path(state.get("sandbox_dir") or ".")
        path = sandbox / PRETEND_LINK_BASENAME
        path.write_text(body + "\n", encoding="utf-8")
        plan_path = sandbox / PRETEND_PLAN_BASENAME
        extra = f"\n\nWrote IR to {path.resolve()}\n"
        if plan_path.is_file():
            extra += f"Plan file: {plan_path.resolve()}\n"
    text = f"PLAN\n----\n{state.get('plan', '')}\n\nIR\n--\n{body}\n{extra}"
    return {"vm_stdout": text, "exit_code": 0}


def _write_pretend_linked(state: PretendState) -> dict:
    """Persist IR to ``pretend-linked.ll`` and clear ``ir_ops`` from state to reduce RAM."""
    sandbox = Path(state.get("sandbox_dir") or ".")
    path = sandbox / PRETEND_LINK_BASENAME
    ops = sanitize_ir_ops(state.get("ir_ops") or [])
    path.write_text(json.dumps(ops, indent=2) + "\n", encoding="utf-8")
    return {
        "ir_ops": [],
        "pretend_linked_path": str(path.resolve()),
    }


def _write_plan_file(state: PretendState) -> dict:
    """Persist precompiler ``plan`` to ``pretend-linked.plan`` (same dir as ``pretend-linked.ll``)."""
    sandbox = Path(state.get("sandbox_dir") or ".")
    path = sandbox / PRETEND_PLAN_BASENAME
    path.write_text(state.get("plan") or "", encoding="utf-8")
    return {"pretend_plan_path": str(path.resolve())}


def _route_after_pre(state: PretendState) -> str:
    if not state.get("validation_ok"):
        return "abort"
    if state.get("use_files"):
        return "write_plan"
    return "compile"


def _route_after_compile(state: PretendState) -> str:
    if not state.get("compile_ok"):
        return "abort"
    if state.get("dry_run"):
        return "dry_run"
    if state.get("use_files"):
        return "write_link"
    return "vm"


def _vm(state: PretendState) -> dict:
    sandbox = state.get("sandbox_dir") or "."
    allow_network = bool(state.get("allow_network"))
    sb_path = Path(sandbox)

    if state.get("use_files"):
        link = sb_path / PRETEND_LINK_BASENAME
        if not link.is_file():
            return {
                "vm_stderr": f"missing IR file: {link}\n(run compile step or disable --use-files)",
                "exit_code": 1,
                "vm_fault_recoverable": False,
                "vm_fault_hint": "",
            }
        try:
            raw = link.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("pretend-linked.ll must contain a JSON array")
            ops: list[dict] = [dict(x) for x in data if isinstance(x, dict)]
        except Exception as e:  # noqa: BLE001
            return {
                "vm_stderr": f"pretend-linked.ll: {e}",
                "exit_code": 1,
                "vm_fault_recoverable": False,
                "vm_fault_hint": "",
            }
    else:
        ops = list(state.get("ir_ops") or [])

    stdout, stderr, code, recoverable, hint = execute_ir(
        ops,
        sandbox_dir=sb_path,
        allow_network=allow_network,
        vm_stdin=str(state.get("vm_stdin") or ""),
        vm_random_seed=state.get("vm_random_seed"),
    )
    out: dict = {
        "vm_stdout": stdout,
        "vm_stderr": stderr,
        "exit_code": code,
        "vm_fault_recoverable": recoverable,
        "vm_fault_hint": hint,
    }
    return out


def _route_after_vm(state: PretendState) -> str:
    retries = int(state.get("retries") or 0)
    max_retries = int(state.get("max_retries") or 2)
    if state.get("vm_fault_recoverable") and retries < max_retries:
        return "retry_compile"
    return "end"


def _bump_retry(state: PretendState) -> dict:
    return {
        "retries": int(state.get("retries") or 0) + 1,
        "vm_fault_recoverable": False,
        "vm_stderr": "",
        "vm_stdout": "",
    }


def build_graph(llm: BaseChatModel):
    g = StateGraph(PretendState)

    g.add_node("precompile", lambda s: run_precompile(llm, s))
    g.add_node("compile", lambda s: run_compile(llm, s))
    g.add_node("abort", _abort)
    g.add_node("dry_run", _dry_run)
    g.add_node("write_plan", _write_plan_file)
    g.add_node("write_link", _write_pretend_linked)
    g.add_node("vm", _vm)
    g.add_node("retry_bump", _bump_retry)

    g.set_entry_point("precompile")

    g.add_conditional_edges(
        "precompile",
        _route_after_pre,
        {"compile": "compile", "abort": "abort", "write_plan": "write_plan"},
    )
    g.add_edge("write_plan", "compile")
    g.add_conditional_edges(
        "compile",
        _route_after_compile,
        {
            "abort": "abort",
            "dry_run": "dry_run",
            "write_link": "write_link",
            "vm": "vm",
        },
    )

    g.add_edge("abort", END)
    g.add_edge("dry_run", END)
    g.add_edge("write_link", "vm")

    g.add_conditional_edges(
        "vm",
        _route_after_vm,
        {"retry_compile": "retry_bump", "end": END},
    )
    g.add_edge("retry_bump", "compile")

    return g


def make_pipeline(llm: BaseChatModel):
    return build_graph(llm).compile()
