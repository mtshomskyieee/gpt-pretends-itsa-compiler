"""pretend-compiler: lowering to fictional IR (`ir_ops`)."""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pretend_compiler.agents.heuristic_ir import (
    heuristic_c_scanf_two_int_divide,
    maybe_fix_under_unrolled_loop,
    synthesize_stdout_ir,
)
from pretend_compiler.agents.ir_normalize import sanitize_ir_ops
from pretend_compiler.agents.schemas import CompilerOut, Diagnostic
from pretend_compiler.structured_kwargs import hosted_structured_output_kwargs
from pretend_compiler.state import PretendState
from pretend_compiler.vm_tool_registry import (
    format_vm_tool_names_for_compiler_prompt,
    format_vm_tool_names_for_fallback_schema,
)

_TOOL_LINE_FALLBACK = (
    '- {{"op":"tool","name":'
    + format_vm_tool_names_for_fallback_schema()
    + ',"args":{{}},"to_stdout":false}}'
)

_SYS = (
    """You are **pretend-compiler**, the middle-end of a pretend compilation pipeline.

Given the **plan**, declared **language**, and **source**, produce a fictional **stack/VM IR** as JSON-compatible
operations in `ir_ops`. Follow the plan’s **iteration counts** and stdout summary when **unrolling** loops: a loop
that runs N times with a `printf` in the body → N separate `print` ops (not one).

Each IR operation is a small JSON object with key `op`. Supported ops:

- `{{"op":"print","text":"..."}}` — pretend program stdout (**include `\\n` at end of each line** so multiple prints do not concatenate)
- `{{"op":"flush_stdout"}}` — no-op; insert **immediately after** the last `print` that matches a C `fflush(stdout)` and **immediately before** the next `read_stdin` (optional but use when the source has `fflush` before `scanf`/`fgets` so the lowering reflects prompt-then-wait)
- `{{"op":"emit_stderr","text":"..."}}` — pretend stderr
- `{{"op":"exit","code":int}}` — terminate with exit code
- `{{"op":"tool","name":<sandbox_tool>,"args":{{...}},"to_stdout":true|false}}`
  **Sandbox tool names (exactly these; no others):**
  """
    + format_vm_tool_names_for_compiler_prompt()
    + """
  Do not use source-language builtins (e.g. Python `range`, `print`, `len`, or C `malloc`) as `name` — they are not tools.
  Use `op":"print"` for all simulated standard output, not a `print` tool.
  Paths must be relative under the sandbox. Common args: `read_file`/`read_bytes`: `path`; `write_file`: `path`, `content`;
  `write_bytes`: `path`, `content_base64` (standard base64); `getenv`: `name`, optional `default`; `http_get`: `url`;
  `http_post`: `url`, optional `body`, `content_type`; `read_stdin`: usually `{{}}` — **one line** from pretend stdin
  (`pretend-run --stdin-text` / `--stdin-file`; empty at EOF). After a `read_stdin`, later `print` ops may embed a **subset**
  of C-style `printf` conversions (`%%`, `%d`, `%u`, `%x`, `%X`, `%o`, `%f`, `%e`, `%g`, `%s`, `%c`) filled from that line’s
  tokens (decimal ints, tokens with `.`/`e` as floats for `%f`, whitespace words for `%s`; if `%f` needs a quotient of two ints,
  the VM uses the first two decimal integers from the line). For `scanf("%d %d", ...)`, one line such as `9 2` with one
  `read_stdin` is enough; two separate `scanf` reads → two `read_stdin` ops in order.
- `{{"op":"fault","message":"...","recoverable":false,"hint":""}}` — fatal VM fault

**Interactive stdin (prompt + read):** Emit `print` (prompts) in **the same order as the source** before the first
`read_stdin` for that read. The VM applies ops **in order** — a prompt line must not appear *after* `read_stdin`
in the IR if it appears *before* `scanf` in the source. After prompt `print`s, optionally `flush_stdout` if the
source has `fflush`, then `read_stdin` for the `scanf`/`fgets`/`input()` line, then the remaining prints.

**IR is a flat list — no branches, no loops, no `for` op.** If the source has `for`/`while`, **unroll** to the right
number of `print` (or other) ops, or unroll a small known iteration count. Do not emit a `tool` to “run” a loop.

**Output must match the real program’s stdout only.** Do not add `print` ops for module docstrings, comments, or
string literals that the source does not pass to `print`/`printf`/`puts`. Do not call `getenv` unless the source
reads the environment: if you do, always pass a non-empty `"name"` in `args` or omit the `getenv` tool.

Behave like a compiler when possible:
- surface duplicate definitions / obvious type clashes as `compile_ok=false` with diagnostics (do NOT emit bad IR).
- if you cannot compile, set `compile_ok=false` and leave `ir_ops` empty.

Return STRICT structured output matching the schema.
"""
)

# Local GGUF models often return empty `ir_ops` with structured binding; plain JSON is more reliable.
_FALLBACK_SYS = (
    """You are **pretend-compiler**. Reply with **ONLY** one JSON object — no markdown fences,
no prose before or after. Valid keys:
- "compile_ok": boolean
- "diagnostics": array of objects with "message" and "severity" (default "error")
- "ir_ops": array of operation objects. Each MUST include "op".

IR ops (same as main compiler):
- {{"op":"print","text":"..."}}
- {{"op":"flush_stdout"}}
- {{"op":"emit_stderr","text":"..."}}
- {{"op":"exit","code":integer}}
- """
    + _TOOL_LINE_FALLBACK
    + """
- {{"op":"fault","message":"...","recoverable":false,"hint":""}}

**Stdin / prompts:** `print` prompts (and optional `flush_stdout` after a fflush) before `read_stdin`; `--stdin-text` or
`--stdin-file` supplies the buffer. `getenv` only with `"name"` in args, or omit it.

**Tool names are ONLY those listed** — never `range`, `print`, `len`, etc. Loops in source unroll to
repeated `print` ops. No docstring/comment text as `print` unless the program prints it.

**Critical:** If compile_ok is true, ir_ops MUST be non-empty. Simulate stdout with one print per line;
each `"text"` MUST end with `\\n`. Do NOT use write_file with path `stdout`, `stderr`, or `-`.
End with {{"op":"exit","code":0}}
unless another exit code is required.
"""
)


def _parse_json_object_from_text(text: str) -> dict | None:
    """Extract the first JSON object from model output (handles optional ```json fences)."""
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_ir_ops(raw: list | None) -> list[dict]:
    out: list[dict] = []
    for x in raw or []:
        if isinstance(x, dict):
            out.append(dict(x))
        elif hasattr(x, "model_dump"):
            out.append(x.model_dump())
    return out


def _compile_json_fallback(
    llm: BaseChatModel,
    *,
    lang: str,
    plan: str,
    source: str,
    hint_block: str,
) -> CompilerOut | None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _FALLBACK_SYS),
            (
                "human",
                "lang={lang}\nplan:\n{plan}\n{hint_block}\nsource:\n```\n{source}\n```\n",
            ),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke(
        {"lang": lang, "plan": plan, "source": source, "hint_block": hint_block}
    )
    content = getattr(msg, "content", None)
    if content is None:
        content = str(msg)
    data = _parse_json_object_from_text(str(content))
    if not data:
        return None
    try:
        return CompilerOut.model_validate(data)
    except Exception:
        return None


def run_compile(llm: BaseChatModel, state: PretendState) -> dict:
    lang = state.get("lang") or ""
    source = state.get("source_text") or ""
    plan = state.get("plan") or ""
    retry_hint = state.get("vm_fault_hint") or ""
    retries = int(state.get("retries") or 0)

    hint_block = ""
    if retries > 0 and retry_hint:
        hint_block = f"\nPrevious VM fault hint (retry {retries}):\n{retry_hint}\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYS),
            (
                "human",
                "lang={lang}\nplan:\n{plan}\n{hint_block}\nsource:\n```\n{source}\n```\n",
            ),
        ]
    )
    so_kw = hosted_structured_output_kwargs(state)
    structured = prompt | llm.with_structured_output(CompilerOut, **so_kw)
    inputs = {"lang": lang, "plan": plan, "source": source, "hint_block": hint_block}
    try:
        out: CompilerOut = structured.invoke(inputs)
    except Exception as exc:  # noqa: BLE001
        out = None
        struct_err = str(exc)
    else:
        struct_err = None

    if out is None:
        fb = _compile_json_fallback(llm, lang=lang, plan=plan, source=source, hint_block=hint_block)
        if fb and _normalize_ir_ops(fb.ir_ops):
            out = fb
        else:
            heur_early = synthesize_stdout_ir(lang, source)
            if heur_early:
                return {
                    "compile_ok": True,
                    "compiler_diagnostics": [
                        {
                            "message": (
                                f"compiler model error ({struct_err}); applied heuristic lowering instead"
                            ),
                            "severity": "warning",
                        }
                    ],
                    "ir_ops": sanitize_ir_ops(heur_early),
                }
            return {
                "compile_ok": False,
                "compiler_diagnostics": [
                    {"message": f"compiler model error: {struct_err}", "severity": "error"}
                ],
                "ir_ops": [],
                "error_message": struct_err,
            }

    ir_ops = _normalize_ir_ops(out.ir_ops)
    diagnostics = list(out.diagnostics)

    if not ir_ops:
        fb = _compile_json_fallback(llm, lang=lang, plan=plan, source=source, hint_block=hint_block)
        if fb:
            cand = _normalize_ir_ops(fb.ir_ops)
            if cand:
                diagnostics = list(out.diagnostics) + list(fb.diagnostics)
                ir_ops = cand
                out = fb

    compile_ok = bool(out.compile_ok)

    fixed, loop_fix = maybe_fix_under_unrolled_loop(lang, source, ir_ops)
    if loop_fix:
        ir_ops = fixed
        compile_ok = True
        diagnostics.append(
            Diagnostic(
                message=(
                    "Substituted heuristic unrolling: model emitted fewer `print` ops than the source loop requires. "
                    "Consider --hosted or a larger model if you need non-trivial lowering."
                ),
                severity="warning",
            )
        )

    # Small local GGUF models often emit no IR — optional deterministic lowering for trivial printf/print loops.
    if not ir_ops:
        heur = synthesize_stdout_ir(lang, source)
        if heur:
            ir_ops = heur
            compile_ok = True
            diagnostics.append(
                Diagnostic(
                    message=(
                        "Applied built-in heuristic lowering (simple for/printf or for/print loop). "
                        "The local model returned no IR; use --hosted or a larger GGUF for real lowering."
                    ),
                    severity="warning",
                )
            )

    if compile_ok and not ir_ops:
        compile_ok = False
        diagnostics.append(
            Diagnostic(
                message=(
                    "compile_ok was true but ir_ops was empty after structured output, JSON fallback, "
                    "and heuristics; try a larger model or --hosted."
                ),
                severity="error",
            )
        )

    # scanf / printf quotient: models often emit linear IR with mutually exclusive branches and multiple exits.
    if (lang or "").strip().lower() == "c":
        div_ir = heuristic_c_scanf_two_int_divide(source)
        if div_ir:
            ir_ops = div_ir
            compile_ok = True
            diagnostics.append(
                Diagnostic(
                    message=(
                        "Applied deterministic scanf→read_stdin→printf lowering for two-int float quotient "
                        "(replaces fragile branchy IR for this pattern)."
                    ),
                    severity="warning",
                )
            )

    ir_ops = sanitize_ir_ops(ir_ops)

    return {
        "compile_ok": compile_ok,
        "compiler_diagnostics": [d.model_dump() for d in diagnostics],
        "ir_ops": ir_ops,
    }
