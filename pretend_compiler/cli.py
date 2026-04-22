"""Typer entrypoints: `pretend-run` and `pretend-models`."""

from __future__ import annotations

import io
import os
import stat
import sys
from pathlib import Path

import typer

from pretend_compiler.graph import make_pipeline
from pretend_compiler.llm_factory import Backend, build_chat_model
from pretend_compiler.models_pull import ollama_pull, pull_gguf
from pretend_compiler.settings import load_settings


def _read_stdin_for_dash() -> str:
    """Handle ``--stdin-file -`` without blocking forever on EOF.

    ``sys.stdin.read()`` waits until EOF. Interactive shells need Enter after one line; pipes and
    shell redirects should slurp the full stream. Some environments report a non-TTY stdin even when
    the user types interactively — those must not use unbounded ``read()`` or the CLI appears hung.
    """
    if sys.stdin.isatty():
        typer.secho("pretend stdin — type one line, then Enter: ", err=True, fg=typer.colors.YELLOW)
        line = sys.stdin.readline()
        return line if line else ""

    try:
        st_mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, io.UnsupportedOperation):
        st_mode = 0

    if stat.S_ISFIFO(st_mode) or stat.S_ISREG(st_mode):
        return sys.stdin.read()

    typer.secho(
        "pretend stdin — type one line, then Enter "
        "(piped input: use `printf '...\\n' | …` or `--stdin-text`): ",
        err=True,
        fg=typer.colors.YELLOW,
        nl=False,
    )
    line = sys.stdin.readline()
    return line if line else ""


def _resolve_vm_stdin(*, stdin_text: str, stdin_file: Path | None) -> str:
    """Text buffer for the VM ``read_stdin`` tool (pretend process stdin)."""
    if stdin_text:
        return stdin_text
    if stdin_file is None:
        return ""
    if str(stdin_file) == "-":
        return _read_stdin_for_dash()
    if not stdin_file.is_file():
        typer.echo(f"--stdin-file: not a file: {stdin_file}", err=True)
        raise typer.Exit(code=2)
    return stdin_file.read_text(encoding="utf-8", errors="replace")

app_models = typer.Typer(add_completion=False, help="Download local model weights")


@app_models.command("pull")
def models_pull(
    name: str = typer.Argument(..., help="mistral | qwen"),
) -> None:
    """Download a pinned GGUF from Hugging Face into the local cache."""
    path, key, env_file = pull_gguf(name)
    typer.echo(f"Downloaded GGUF to:\n  {path}")
    typer.echo(f"Updated {env_file}:\n  {key}={path}")


@app_models.command("ollama")
def models_ollama(
    model: str = typer.Argument("mistral", help="Model tag passed to `ollama pull`"),
) -> None:
    """Pull weights via the Ollama CLI (optional backend)."""
    ollama_pull(model)
    typer.echo(f"ollama pull {model} completed.")


def main_models() -> None:
    app_models()


def run_entry(
    lang: str = typer.Option(
        ...,
        "--lang",
        help="Declared source language (python, c, cpp, rust, ...)",
    ),
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    hosted: bool = typer.Option(
        False,
        "--hosted",
        help="Use OpenAI-compatible API from llm.env (ignores local --backend)",
    ),
    backend: str = typer.Option(
        "mistral",
        "--backend",
        help="Local backend when not using --hosted: mistral | qwen | ollama",
    ),
    sandbox_dir: Path = typer.Option(
        Path.cwd(),
        "--sandbox-dir",
        file_okay=False,
        help="Sandbox root for VM file tools",
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help="Allow http_get/http_post in VM tools",
    ),
    vm_random_seed: int | None = typer.Option(
        None,
        "--vm-random-seed",
        help="Seed for VM random_int/random_float tools (deterministic pretend runs)",
    ),
    max_retries: int = typer.Option(
        2,
        "--max-retries",
        min=0,
        max=10,
        help="VM→compiler retries on recoverable VM faults",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Stop after compiler: print plan + IR only",
    ),
    use_files: bool = typer.Option(
        False,
        "--use-files",
        help=(
            "Write pretend-linked.plan after validation and pretend-linked.ll after compile; "
            "run VM from the .ll file (reduces large strings / ir_ops in pipeline state)"
        ),
    ),
    stdin_text: str = typer.Option(
        "",
        "--stdin-text",
        help="Pretend process stdin for the VM read_stdin tool (overrides --stdin-file if set)",
    ),
    stdin_file: Path | None = typer.Option(
        None,
        "--stdin-file",
        exists=False,
        file_okay=True,
        dir_okay=False,
        help=(
            "Pretend stdin from file, or '-' — piped stdin reads until EOF; interactive terminal reads "
            "one line after Enter (see --stdin-text for a literal). Ignored if --stdin-text is set"
        ),
    ),
) -> None:
    settings = load_settings()
    bk_lower = backend.lower().strip()
    if bk_lower not in ("mistral", "qwen", "ollama"):
        typer.echo(
            f"Unknown --backend {backend!r}; expected mistral|qwen|ollama",
            err=True,
        )
        raise typer.Exit(code=2)

    bk: Backend = bk_lower  # type: ignore[assignment]

    try:
        llm = build_chat_model(settings=settings, hosted=hosted, backend=bk)
    except Exception as e:  # noqa: BLE001
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    source_text = path.read_text(encoding="utf-8", errors="replace")
    vm_stdin = _resolve_vm_stdin(stdin_text=stdin_text, stdin_file=stdin_file)
    init: dict = {
        "lang": lang,
        "source_path": str(path.resolve()),
        "source_text": source_text,
        "hosted": hosted,
        "backend": bk_lower,
        "allow_network": allow_network,
        "sandbox_dir": str(sandbox_dir.resolve()),
        "vm_stdin": vm_stdin,
        "vm_random_seed": vm_random_seed,
        "max_retries": max_retries,
        "retries": 0,
        "dry_run": dry_run,
        "use_files": use_files,
        "validation_diagnostics": [],
        "compiler_diagnostics": [],
        "ir_ops": [],
    }

    pipe = make_pipeline(llm)
    try:
        out = pipe.invoke(init)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"pipeline error: {e}", err=True)
        raise typer.Exit(code=1)

    stdout = out.get("vm_stdout") or ""
    stderr = out.get("vm_stderr") or ""
    code = int(out.get("exit_code") or 0)

    if stdout:
        sys.stdout.write(stdout if stdout.endswith("\n") else stdout + "\n")
    if stderr:
        sys.stderr.write(stderr if stderr.endswith("\n") else stderr + "\n")
    if (
        code == 0
        and not dry_run
        and not stdout
        and not stderr
    ):
        typer.echo(
            "(Pretend VM exited 0 with no stdout/stderr. Try `--dry-run` to print plan + IR, "
            "or set LLM_MAX_TOKENS=8192 in llm.env if responses were truncated.)",
            err=True,
        )
    raise typer.Exit(code=code)


def main_run() -> None:
    """Console script entry for `pretend-run`."""
    typer.run(run_entry)


if __name__ == "__main__":
    main_run()
