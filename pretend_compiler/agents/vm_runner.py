"""Execute compiler-emitted IR ops with sandboxed tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pretend_compiler.agents.ir_normalize import sanitize_ir_ops
from pretend_compiler.agents.vm_tools import make_vm_tools, run_tool_by_name


def _split_words(line: str) -> list[str]:
    return [w for w in line.replace(",", " ").split() if w]


def _try_int_token(tok: str) -> int | None:
    try:
        return int(tok, 10)
    except ValueError:
        return None


@dataclass
class _FormatCtx:
    """Sequential stdin tokens for scanf-like ``printf`` expansion after ``read_stdin``."""

    tokens: list[str]
    ti: int = 0

    def take_int(self) -> int | None:
        if self.ti >= len(self.tokens):
            return None
        v = _try_int_token(self.tokens[self.ti])
        if v is None:
            return None
        self.ti += 1
        return v

    def take_float(self) -> float | None:
        if self.ti >= len(self.tokens):
            return None
        t0 = self.tokens[self.ti]
        if "." in t0 or "e" in t0.lower():
            try:
                v = float(t0)
            except ValueError:
                return None
            self.ti += 1
            return v
        if self.ti + 1 < len(self.tokens):
            a = _try_int_token(t0)
            b = _try_int_token(self.tokens[self.ti + 1])
            if a is not None and b is not None and b != 0:
                self.ti += 2
                return float(a) / float(b)
        iv = _try_int_token(t0)
        if iv is not None:
            self.ti += 1
            return float(iv)
        return None

    def take_str(self) -> str | None:
        if self.ti >= len(self.tokens):
            return None
        w = self.tokens[self.ti]
        self.ti += 1
        return w


def _parse_printf_spec(text: str, start: int) -> tuple[dict[str, Any], int] | None:
    """
    Parse one C-style conversion starting at ``text[start] == '%'``.
    Returns (spec_dict, index_after_spec) or None if not a valid spec.
    """
    if start >= len(text) or text[start] != "%":
        return None
    i = start + 1
    if i < len(text) and text[i] == "%":
        return ({"conv": "%"}, start + 2)

    flags = ""
    while i < len(text) and text[i] in "-+ #0":
        flags += text[i]
        i += 1

    width = None
    if i < len(text) and text[i].isdigit():
        j = i
        while j < len(text) and text[j].isdigit():
            j += 1
        width = int(text[i:j], 10)
        i = j

    prec = None
    if i < len(text) and text[i] == ".":
        i += 1
        j = i
        while j < len(text) and text[j].isdigit():
            j += 1
        prec = int(text[i:j], 10) if j > i else 0
        i = j

    while i < len(text):
        if text[i : i + 2] == "ll":
            i += 2
            continue
        if text[i] in "hlLjzt":
            i += 1
            continue
        break

    if i >= len(text):
        return None
    conv_ch = text[i]
    conv = conv_ch.lower()
    if conv not in "diuxxfeeggcsp":
        return None
    i += 1
    return (
        {
            "conv": conv,
            "conv_ch": conv_ch,
            "flags": flags,
            "width": width,
            "precision": prec,
        },
        i,
    )


def _format_int_conv(
    conv: str,
    value: int,
    *,
    flags: str,
    width: int | None,
    precision: int | None,
) -> str:
    unsigned = conv in ("u", "x", "X", "o")
    base = 10
    prefix = ""
    upper_hex = False
    if conv == "x":
        base = 16
        if "#" in flags:
            prefix = "0x"
        upper_hex = False
    elif conv == "X":
        base = 16
        if "#" in flags:
            prefix = "0x"
        upper_hex = True
    elif conv == "o":
        base = 8
        if "#" in flags and value != 0:
            prefix = "0"

    if unsigned:
        if conv == "u":
            v = value % (2**32) if value < 0 else value
            s = str(v % (2**32))
        elif conv in ("x", "X"):
            u = value % (2**32)
            xs = format(u, "x")
            if upper_hex:
                xs = xs.upper()
            s = prefix + xs if prefix else xs
        else:  # o
            u = value % (2**32)
            o = format(u, "o")
            s = prefix + o if prefix else o
    else:
        s = str(value)
        if precision is not None and precision > len(s.removeprefix("-")):
            pad0 = precision - (len(s) - 1 if s.startswith("-") else len(s))
            if s.startswith("-"):
                body = s[1:].zfill(max(0, precision - 1))
                s = "-" + body
            else:
                s = s.zfill(precision)

    if conv in ("d", "i"):
        if value >= 0:
            if "+" in flags:
                s = "+" + s
            elif " " in flags:
                s = " " + s

    if width is not None and len(s) < width:
        pad_ch = "0" if ("0" in flags and "-" not in flags and conv in "diuxXo") else " "
        if "-" in flags:
            s = s + pad_ch * (width - len(s))
        else:
            s = pad_ch * (width - len(s)) + s
    return s


def _format_float_conv(
    value: float,
    *,
    flags: str,
    width: int | None,
    precision: int | None,
    conv: str,
) -> str:
    prec = 6 if precision is None else precision
    if conv in "fe":
        body = f"{value:.{prec}f}"
    elif conv == "g":
        body = f"{value:.{prec}g}"
    else:
        body = f"{value:.{prec}f}"

    if value >= 0:
        if "+" in flags:
            body = "+" + body
        elif " " in flags:
            body = " " + body
    if width is not None and len(body) < width:
        pad = " " if "-" in flags or "0" not in flags else "0"
        if "-" in flags:
            body = body + pad * (width - len(body))
        else:
            body = pad * (width - len(body)) + body
    return body


def _expand_printf_text(text: str, ctx: _FormatCtx | None) -> str:
    """
    Substitute C-style printf specifiers using ``ctx`` (stdin-derived queues).
    Supported: %%, %d %i %u %x %X %o %f %e %g %c %s (partial flags/width/precision).
    """
    if not ctx or "%" not in text:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "%":
            out.append(text[i])
            i += 1
            continue
        parsed = _parse_printf_spec(text, i)
        if parsed is None:
            out.append(text[i])
            i += 1
            continue
        spec, end = parsed
        conv = spec["conv"]
        if conv == "%":
            out.append("%")
            i = end
            continue

        flags = spec["flags"]
        width = spec["width"]
        precision = spec["precision"]

        if conv in "diuxo":
            iv = ctx.take_int()
            if iv is None:
                iv = 0
            iconv = str(spec.get("conv_ch") or conv)
            piece = _format_int_conv(
                iconv,
                iv,
                flags=flags,
                width=width,
                precision=precision,
            )
        elif conv in "feg":
            fv = ctx.take_float()
            if fv is None:
                fv = 0.0
            piece = _format_float_conv(
                fv,
                flags=flags,
                width=width,
                precision=precision,
                conv=conv,
            )
        elif conv == "s":
            sv = ctx.take_str()
            if sv is None:
                sv = ""
            piece = sv
            if precision is not None:
                piece = piece[:precision]
            if width is not None and len(piece) < width:
                pad = " " * (width - len(piece))
                piece = piece + pad if "-" in flags else pad + piece
        elif conv == "c":
            w = ctx.take_str()
            if w is None:
                iv = ctx.take_int()
                ch = chr(iv % 256) if iv is not None else "?"
            else:
                ch = w[0] if w else ""
            piece = ch
        else:
            piece = text[i:end]
        out.append(piece)
        i = end
    return "".join(out)


def execute_ir(
    ir_ops: list[dict[str, Any]],
    *,
    sandbox_dir: Path,
    allow_network: bool,
    vm_stdin: str = "",
    vm_random_seed: int | None = None,
) -> tuple[str, str, int, bool, str]:
    """Run IR sequentially. Returns stdout, stderr, exit_code, fault_recoverable, fault_hint.

    ``vm_stdin`` is a text buffer consumed line-by-line by the ``read_stdin`` tool (pretend process stdin).
    ``vm_random_seed`` seeds RNG tools for reproducible runs (optional).
    """

    ir_ops = sanitize_ir_ops(ir_ops)

    tools = make_vm_tools(
        sandbox_dir,
        allow_network=allow_network,
        stdin_text=vm_stdin,
        random_seed=vm_random_seed,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    format_ctx: _FormatCtx | None = None

    for i, op in enumerate(ir_ops):
        kind = (op or {}).get("op")
        if kind == "print":
            text = str(op.get("text", ""))
            if format_ctx is not None and "%" in text:
                text = _expand_printf_text(text, format_ctx)
            stdout_parts.append(text)
        elif kind == "flush_stdout":
            # No-op: documents prompt-before-input ordering (C fflush); VM already runs ops sequentially.
            pass
        elif kind == "emit_stderr":
            stderr_parts.append(str(op.get("text", "")))
        elif kind == "exit":
            code = int(op.get("code", 0))
            return "".join(stdout_parts), "".join(stderr_parts), code, False, ""
        elif kind == "tool":
            name = str(op.get("name", ""))
            args = dict(op.get("args") or {})
            out = run_tool_by_name(tools, name, args)
            if name == "read_stdin":
                line = out.rstrip("\n") if isinstance(out, str) else ""
                format_ctx = _FormatCtx(tokens=_split_words(line))
            if op.get("to_stdout"):
                stdout_parts.append(out)
            else:
                stdout_parts.append(out + ("\n" if out and not out.endswith("\n") else ""))
        elif kind == "fault":
            msg = str(op.get("message", "fault"))
            stderr_parts.append(msg)
            recoverable = bool(op.get("recoverable", False))
            hint = str(op.get("hint", ""))
            return (
                "".join(stdout_parts),
                "".join(stderr_parts),
                1,
                recoverable,
                hint,
            )
        else:
            stderr_parts.append(f"unknown IR op at index {i}: {op!r}")
            return "".join(stdout_parts), "".join(stderr_parts), 1, False, ""

    return "".join(stdout_parts), "".join(stderr_parts), 0, False, ""
