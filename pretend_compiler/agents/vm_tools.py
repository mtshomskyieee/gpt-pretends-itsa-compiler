"""Sandboxed tools invoked by the pretend VM when executing IR."""

from __future__ import annotations

import base64
import io
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from pretend_compiler.vm_tool_registry import VM_SANDBOX_TOOL_NAMES

_MAX_HTTP_BYTES = 2_000_000
_MAX_BINARY_BYTES = 2_000_000
_LIST_DIR_CAP = 5_000
_SLEEP_MS_MAX = 2_000


def _ensure_under_sandbox(sandbox_root: Path, rel_or_abs: str) -> Path:
    root = sandbox_root.resolve()
    candidate = (root / rel_or_abs).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Path escapes sandbox: {rel_or_abs!r}") from e
    return candidate


def make_vm_tools(
    sandbox_root: Path,
    *,
    allow_network: bool,
    stdin_text: str = "",
    random_seed: int | None = None,
) -> list[StructuredTool]:
    """Build LangChain tools for file, env, optional HTTP, pretend stdin, time, and RNG."""

    stdin_buf = io.StringIO(stdin_text if stdin_text is not None else "")
    rng = random.Random(random_seed if random_seed is not None else time.time_ns())

    def read_file(path: str) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        if not p.is_file():
            return f"[tool error] not a file: {path}"
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[tool error] read failed: {e}"

    def write_file(path: str, content: str) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(content, encoding="utf-8")
            return f"ok wrote {len(content)} bytes to {path}"
        except OSError as e:
            return f"[tool error] write failed: {e}"

    def read_bytes(path: str) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        if not p.is_file():
            return f"[tool error] not a file: {path}"
        try:
            raw = p.read_bytes()
        except OSError as e:
            return f"[tool error] read failed: {e}"
        if len(raw) > _MAX_BINARY_BYTES:
            return f"[tool error] file too large ({len(raw)} bytes; max {_MAX_BINARY_BYTES})"
        return base64.b64encode(raw).decode("ascii")

    def write_bytes(path: str, content_base64: str) -> str:
        try:
            raw = base64.b64decode(content_base64.strip(), validate=True)
        except (ValueError, TypeError) as e:
            return f"[tool error] invalid base64: {e}"
        if len(raw) > _MAX_BINARY_BYTES:
            return f"[tool error] payload too large ({len(raw)} bytes; max {_MAX_BINARY_BYTES})"
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_bytes(raw)
            return f"ok wrote {len(raw)} bytes to {path}"
        except OSError as e:
            return f"[tool error] write failed: {e}"

    def getenv(name: str = "", default: str = "") -> str:
        return os.environ.get(name, default) if name else default

    def http_get(url: str, timeout_sec: float = 10.0) -> str:
        if not allow_network:
            return "[tool error] network disabled; pass --allow-network to enable http_get"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "pretend-compiler-vm/0.1"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read(_MAX_HTTP_BYTES)
                status = getattr(resp, "status", None) or resp.getcode()
                ct = resp.headers.get("Content-Type", "")
                text = body.decode("utf-8", errors="replace")
                return f"HTTP/{status}\nContent-Type: {ct}\n\n{text}"
        except urllib.error.HTTPError as e:
            return f"[tool error] HTTP {e.code}: {e.reason}"
        except Exception as e:
            return f"[tool error] {type(e).__name__}: {e}"

    def http_post(
        url: str,
        body: str = "",
        content_type: str = "application/octet-stream",
        timeout_sec: float = 10.0,
    ) -> str:
        if not allow_network:
            return "[tool error] network disabled; pass --allow-network to enable http_post"
        try:
            data = body.encode("utf-8") if body else None
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "User-Agent": "pretend-compiler-vm/0.1",
                    "Content-Type": content_type,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                out = resp.read(_MAX_HTTP_BYTES)
                status = getattr(resp, "status", None) or resp.getcode()
                ct = resp.headers.get("Content-Type", "")
                text = out.decode("utf-8", errors="replace")
                return f"HTTP/{status}\nContent-Type: {ct}\n\n{text}"
        except urllib.error.HTTPError as e:
            body_err = (
                e.read(_MAX_HTTP_BYTES).decode("utf-8", errors="replace") if e.fp else ""
            )
            return f"[tool error] HTTP {e.code}: {e.reason}\n{body_err}"
        except Exception as e:
            return f"[tool error] {type(e).__name__}: {e}"

    def read_stdin() -> str:
        """One line of pretend standard input (injected for this VM run via the CLI)."""
        return stdin_buf.readline()

    def list_dir(path: str = ".", recursive: bool = False) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        if not p.is_dir():
            return f"[tool error] not a directory: {path}"
        names: list[str] = []
        try:
            if recursive:
                for root, dirs, files in os.walk(p, topdown=True):
                    base = Path(root).relative_to(p)
                    for d in dirs:
                        rel = str(base / d).replace("\\", "/")
                        if rel != ".":
                            names.append(rel + "/")
                    for f in files:
                        rel = str(base / f).replace("\\", "/")
                        names.append(rel)
                    if len(names) >= _LIST_DIR_CAP:
                        break
            else:
                for child in sorted(p.iterdir()):
                    names.append(child.name + ("/" if child.is_dir() else ""))
        except OSError as e:
            return f"[tool error] list failed: {e}"
        if len(names) >= _LIST_DIR_CAP:
            names = names[:_LIST_DIR_CAP]
            names.append("[truncated]")
        return "\n".join(names)

    def path_exists(path: str) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        return "yes" if p.exists() else "no"

    def remove_file(path: str) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        if not p.is_file():
            return f"[tool error] not a file: {path}"
        try:
            p.unlink()
            return f"ok removed {path}"
        except OSError as e:
            return f"[tool error] remove failed: {e}"

    def rename(from_path: str, to_path: str) -> str:
        try:
            src = _ensure_under_sandbox(sandbox_root, from_path)
            dst = _ensure_under_sandbox(sandbox_root, to_path)
        except ValueError as e:
            return f"[tool error] {e}"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            return f"ok renamed {from_path} -> {to_path}"
        except OSError as e:
            return f"[tool error] rename failed: {e}"

    def mkdir(path: str, parents: bool = True) -> str:
        try:
            p = _ensure_under_sandbox(sandbox_root, path)
        except ValueError as e:
            return f"[tool error] {e}"
        try:
            p.mkdir(parents=parents, exist_ok=True)
            return f"ok mkdir {path}"
        except OSError as e:
            return f"[tool error] mkdir failed: {e}"

    def sleep_ms(ms: int) -> str:
        n = max(0, min(int(ms), _SLEEP_MS_MAX))
        time.sleep(n / 1000.0)
        return f"ok slept {n} ms"

    def time_now() -> str:
        epoch_ms = int(time.time() * 1000)
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return json.dumps({"epoch_ms": epoch_ms, "iso_utc": iso})

    def random_int(low: int = 0, high: int = 2_147_483_647) -> str:
        lo, hi = int(low), int(high)
        if hi <= lo:
            return f"[tool error] invalid range: low={lo} high={hi}"
        return str(rng.randrange(lo, hi))

    def random_float() -> str:
        return repr(rng.random())

    tools: list[StructuredTool] = [
        StructuredTool.from_function(read_file, name="read_file", description="Read a UTF-8 text file under the sandbox."),
        StructuredTool.from_function(write_file, name="write_file", description="Write UTF-8 text to a path under the sandbox."),
        StructuredTool.from_function(read_bytes, name="read_bytes", description="Read a binary file; returns standard base64 (max 2MiB)."),
        StructuredTool.from_function(write_bytes, name="write_bytes", description="Write binary data from standard base64 (max 2MiB decoded)."),
        StructuredTool.from_function(getenv, name="getenv", description="Read an environment variable (process env)."),
        StructuredTool.from_function(http_get, name="http_get", description="HTTP GET URL (requires --allow-network). Returns status line, Content-Type, blank line, body."),
        StructuredTool.from_function(http_post, name="http_post", description="HTTP POST with UTF-8 body (requires --allow-network). Args: url, body, content_type."),
        StructuredTool.from_function(
            read_stdin,
            name="read_stdin",
            description=(
                "Read one line from pretend stdin (set with pretend-run --stdin-text, --stdin-file, or --stdin-file -). "
                "Empty string at EOF. Use for sources that read with scanf, input(), etc."
            ),
        ),
        StructuredTool.from_function(list_dir, name="list_dir", description="List names under a sandbox directory (optional recursive; capped)."),
        StructuredTool.from_function(path_exists, name="path_exists", description="Returns yes/no whether a sandbox path exists."),
        StructuredTool.from_function(remove_file, name="remove_file", description="Delete a file under the sandbox."),
        StructuredTool.from_function(rename, name="rename", description="Rename/move from_path to_path under the sandbox."),
        StructuredTool.from_function(mkdir, name="mkdir", description="Create a directory under the sandbox (parents=True by default)."),
        StructuredTool.from_function(sleep_ms, name="sleep_ms", description=f"Sleep up to {_SLEEP_MS_MAX} ms (wall clock)."),
        StructuredTool.from_function(time_now, name="time_now", description="Current time as JSON: epoch_ms and iso_utc (UTC)."),
        StructuredTool.from_function(random_int, name="random_int", description="Random integer in [low, high). Seeded when VM sets vm_random_seed."),
        StructuredTool.from_function(random_float, name="random_float", description="Random float in [0,1). Seeded when VM sets vm_random_seed."),
    ]
    registered = {t.name for t in tools}
    if registered != VM_SANDBOX_TOOL_NAMES:
        missing = sorted(VM_SANDBOX_TOOL_NAMES - registered)
        extra = sorted(registered - VM_SANDBOX_TOOL_NAMES)
        raise RuntimeError(
            "VM tool names drifted from vm_tool_registry.VM_SANDBOX_TOOL_NAMES: "
            f"missing={missing!r} extra={extra!r}"
        )
    return tools


def run_tool_by_name(
    tools: list[StructuredTool],
    name: str,
    args: dict[str, Any],
) -> str:
    for t in tools:
        if t.name == name:
            try:
                return str(t.invoke(args))
            except Exception as e:  # noqa: BLE001
                return f"[tool error] {type(e).__name__}: {e}"
    return f"[tool error] unknown tool: {name}"
