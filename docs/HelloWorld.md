# HelloWorld: `pretend-run` orchestration walkthrough

This document walks through what happens when you run the pretend compiler against the sample C program, from the CLI entrypoint through LangGraph, the two LLM agents, optional on-disk artifacts, and the deterministic VM.

## Command (and the `--use-files` flag)

Use this command from the repo root (after `uv sync` and configuring hosted API credentials, see below):

```bash
uv run pretend-run --hosted --use-files --lang c sample.c
```

The CLI exposes **`--use-files`**, not `--using-files`. If you are searching docs for "using files," use **`--use-files`** so copy-paste works.

## Prerequisites

- **Dependencies**: Install with `uv sync` (see [README.md](../README.md)).
- **`--hosted`**: Uses an OpenAI-compatible HTTP API via LangChain `ChatOpenAI`. You need **`OPENAI_API_KEY`** in `llm.env` (or the environment). Optional variables include `OPENAI_BASE_URL` and `OPENAI_MODEL`. See [pretend_compiler/llm_factory.py](../pretend_compiler/llm_factory.py) and [pretend_compiler/settings.py](../pretend_compiler/settings.py). When hosted is enabled, **`--backend`** (local mistral/qwen/ollama) does not choose the provider.
- **`--lang c`**: Declares the source language; it becomes `lang` in pipeline state and drives precompiler/compiler prompts plus C-specific heuristics in the compiler.
- **`sample.c`**: A tiny program that loops ten times printing `hello world` ([sample.c](../sample.c)). The pretend pipeline does **not** call `gcc`; it lowers the source to fictional IR and interprets that IR in a sandbox VM.

## Entry: from script to graph

1. **Console script**: [pyproject.toml](../pyproject.toml) maps `pretend-run` to `pretend_compiler.cli:main_run`, which calls `typer.run(run_entry)` in [pretend_compiler/cli.py](../pretend_compiler/cli.py).
2. **Settings**: `load_settings()` reads **`llm.env`** (among other sources) into a `Settings` object.
3. **Model**: `build_chat_model(settings=..., hosted=True, backend=...)` returns the hosted chat model when `--hosted` is set.
4. **Source**: `run_entry` reads **`sample.c`** into `source_text`, stores `source_path`, and builds the initial graph state (`lang`, `hosted`, `use_files: true`, `sandbox_dir`, empty `ir_ops`, and related fields). Default **`--sandbox-dir`** is the current working directory.
5. **Pipeline**: `make_pipeline(llm)` compiles the LangGraph defined in [pretend_compiler/graph.py](../pretend_compiler/graph.py); `pipe.invoke(init)` runs from node **`precompile`** through conditional edges to **`END`** (or **`abort`**).
6. **Process output**: Whatever ends up in `vm_stdout` / `vm_stderr` is written to the real stdout/stderr; the process exits with **`exit_code`** from the VM (or abort). If exit code is 0 and there is no stdout or stderr, the CLI may print a hint about `--dry-run` or `LLM_MAX_TOKENS` ([cli.py](../pretend_compiler/cli.py)).

Initial state shape is summarized by [`PretendState`](../pretend_compiler/state.py).

## LangGraph path for `--hosted --use-files --lang c sample.c`

With **`--use-files`** and **without** `--dry-run`, the graph follows this order (mirroring [pretend_compiler/graph.py](../pretend_compiler/graph.py)):

1. **`precompile`** - [`run_precompile`](../pretend_compiler/agents/precompiler.py) runs a lightweight deterministic check for C (brace balance among other rules) and calls the LLM with structured output. It sets **`validation_ok`**, **`plan`** (only if validation passes), and **`validation_diagnostics`**.
2. **Route after precompile** - If validation fails, go to **`abort`**. If validation succeeds and **`use_files`** is true, go to **`write_plan`** instead of straight to **`compile`**.
3. **`write_plan`** - Writes the precompiler's **`plan`** text to **`{sandbox_dir}/pretend-linked.plan`** and records **`pretend_plan_path`**. Then always edges to **`compile`**.
4. **`compile`** - [`run_compile`](../pretend_compiler/agents/compiler.py) lowers the source to **`ir_ops`** using structured output, optional plain-JSON fallback, loop fixes, built-in heuristics for trivial printf/print loops when the model returns empty IR, and optional C scanf patterns. Results include **`compile_ok`** and **`compiler_diagnostics`**; ops pass through **`sanitize_ir_ops`**.
5. **Route after compile** - If **`compile_ok`** is false -> **`abort`**. If **`dry_run`** -> **`dry_run`** node (print plan + IR; with `--use-files`, also writes `pretend-linked.ll` under the sandbox, see README). If compile succeeded and **`use_files`** -> **`write_link`**. Otherwise -> **`vm`** with IR kept in memory.
6. **`write_link`** - Writes sanitized IR JSON to **`{sandbox_dir}/pretend-linked.ll`** and clears **`ir_ops`** in state so large IR lists are not held in RAM. Then edges to **`vm`**.
7. **`vm`** - When **`use_files`** is true, IR is **read from disk** (`pretend-linked.ll`), parsed as a JSON array of op objects. [`execute_ir`](../pretend_compiler/agents/vm_runner.py) runs those ops with **`sandbox_dir`**, **`allow_network`**, **`vm_stdin`**, and **`vm_random_seed`** as configured in state. Emitted **`vm_stdout`**, **`vm_stderr`**, **`exit_code`**, and optional fault fields are written back into state.
8. **Route after VM** - If the VM reports a **`vm_fault_recoverable`** fault and **`retries` < `max_retries`** (default **2**), go to **`retry_bump`** (increment retries, clear prior VM outputs), then back to **`compile`** with **`vm_fault_hint`** available for the compiler prompt. Otherwise **`END`**.

**Abort**: The **`abort`** node aggregates validation or compiler diagnostics into **`vm_stderr`** and sets **`exit_code`** to **1**, then **`END`**.

```mermaid
flowchart TB
    start([entry]) --> precompile
    precompile[["precompile"]] -->|validation fails| abort
    precompile -->|"ok + --use-files"| write_plan
    write_plan[["write_plan"]] --> compile
    compile[["compile"]] -->|"not compile_ok"| abort
    compile -->|"compile ok + --use-files"| write_link
    write_link[["write_link"]] --> vm
    vm[["vm"]] -->|"recoverable fault, retries left"| retry_bump
    vm --> end([END])
    retry_bump[retry_bump] --> compile
    abort[abort] --> end
```

For a fuller diagram including **`dry_run`** and in-memory **`vm`** paths, see the flowchart in [README.md](../README.md).

## What "running" means here

The **VM** does not execute machine code or shell out to **`gcc`**. It executes a linear list of fictional operations, such as **`print`**, **`tool`** (sandbox helpers), **`exit`**, etc., defined by the compiler agent and interpreted in [pretend_compiler/agents/vm_runner.py](../pretend_compiler/agents/vm_runner.py) with tools from [pretend_compiler/agents/vm_tools.py](../pretend_compiler/agents/vm_tools.py). For [sample.c](../sample.c), the observable goal is stdout that resembles ten lines of `hello world`, produced by whatever IR the model and heuristics emit, not by running the real `main()` binary.

## Artifacts and defaults

- With **`--use-files`**, expect **`pretend-linked.plan`** (after successful validation) and **`pretend-linked.ll`** (after successful compile, before VM) under **`--sandbox-dir`**. By default that is **`.`**, so both files appear in the current working directory unless you pass e.g. **`--sandbox-dir /path/to/dir`**.
- **`--dry-run`** stops after compile: the **`dry_run`** node prints **PLAN** and **IR** to the pretend stdout channel and skips VM execution; with **`--use-files`**, **`pretend-linked.ll`** can still be written, see **`_dry_run`** in [pretend_compiler/graph.py](../pretend_compiler/graph.py).

## Where to read more

- Overview and usage: [README.md](../README.md)
- IR, backends, and sandbox rules in depth: [ARCHITECTURE.md](ARCHITECTURE.md)

This walkthrough does not enumerate every IR opcode; use those documents for reference.
