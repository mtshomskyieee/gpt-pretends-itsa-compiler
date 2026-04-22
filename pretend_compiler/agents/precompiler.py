"""pretend-precompiler: language conformance gate + plan."""

from __future__ import annotations

import ast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pretend_compiler.agents.schemas import Diagnostic, PrecompilerOut
from pretend_compiler.structured_kwargs import hosted_structured_output_kwargs
from pretend_compiler.state import PretendState


_SYS = """You are **pretend-precompiler**, the front-end of a pretend compiler.

Your job has TWO parts:

1) **Conformance gate** for the DECLARED language (`lang`). Decide if the source plausibly matches
   that language's surface syntax and obvious rules. Examples:
   - **python**: indentation structure, colons after compound statements, unmatched brackets.
   - **c**, **cpp**, **c++**: braces, semicolons after statements (be reasonable about macros), basic balance.
   - **java**: class structure, semicolons, braces.
   - **rust**: `fn`, `let`, `mut`, basic module/use plausibility.

   If the source is clearly the WRONG language for `lang`, set validation_ok=false.

2) **Plan** (only meaningful if validation_ok=true): brief instructions for **pretend-compiler**, the next stage.
   That backend emits a **flat** IR (no loop op, no branches): every `for`/`while` must become **unrolled**
   repeated ops (e.g. ten iterations → ten `print`s). Your plan MUST therefore:
   - List each loop with its **iteration count** (or exact bound, e.g. “`i < 10` → 10 iterations”).
   - Note **variables** that affect stdout, stdin, or control flow so lowering preserves them.
   - State how body side effects map to IR (especially repeated `printf`/`print` lines).
   - Summarize expected **stdout** when relevant (how many lines, order).

Return STRICT structured fields only (via the schema). Diagnostics must use human-readable messages.
"""


def _deterministic_python(source: str) -> tuple[bool, list[Diagnostic]]:
    try:
        ast.parse(source)
        return True, []
    except SyntaxError as e:
        d = Diagnostic(
            line=e.lineno,
            column=e.offset,
            message=str(e.msg or e),
            severity="error",
        )
        return False, [d]


def _deterministic_brace_balance(lang: str, source: str) -> tuple[bool, list[Diagnostic]]:
    """Lightweight brace balance for C-family; LLM remains authoritative for finer rules."""

    lang_l = lang.lower().strip()
    if lang_l not in {"c", "cpp", "c++", "cxx", "java"}:
        return True, []

    stripped = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    opens = stripped.count("{")
    closes = stripped.count("}")
    if opens != closes:
        return False, [
            Diagnostic(message="Unbalanced braces `{` `}` (heuristic check).", severity="error")
        ]
    return True, []


def run_precompile(llm: BaseChatModel, state: PretendState) -> dict:
    lang = state.get("lang") or ""
    source = state.get("source_text") or ""
    path = state.get("source_path") or ""

    det_ok = True
    det_diag: list[Diagnostic] = []
    if lang.lower().strip() == "python":
        det_ok, det_diag = _deterministic_python(source)
    else:
        ok_b, diag_b = _deterministic_brace_balance(lang, source)
        if not ok_b:
            det_ok = False
            det_diag.extend(diag_b)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYS),
            (
                "human",
                "lang={lang}\npath={path}\n\nsource:\n```\n{source}\n```\n",
            ),
        ]
    )
    so_kw = hosted_structured_output_kwargs(state)
    structured = prompt | llm.with_structured_output(PrecompilerOut, **so_kw)
    try:
        out: PrecompilerOut = structured.invoke(
            {"lang": lang, "path": path, "source": source}
        )
    except Exception as exc:  # noqa: BLE001 — surface model errors
        return {
            "validation_ok": False,
            "validation_diagnostics": [{"message": f"precompiler model error: {exc}", "severity": "error"}],
            "plan": "",
            "error_message": str(exc),
        }

    merged_diag = [d.model_dump() for d in det_diag] + [d.model_dump() for d in out.diagnostics]
    validation_ok = bool(det_ok and out.validation_ok)

    return {
        "validation_ok": validation_ok,
        "validation_diagnostics": merged_diag,
        "plan": out.plan if validation_ok else "",
    }
