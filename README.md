# Pretend compiler (uv + LangGraph)

CLI that **pretends** to compile and run code in a language you choose, using three LLM-driven stages: **precompiler** (language gate + plan), **compiler** (lowering to a small IR), **VM** (interprets IR with optional sandboxed file/network tools).

**IR** stands for **intermediate representation**: in this project it is the fictional lowering target—not machine code—a JSON array of operation objects (`print`, `tool`, `exit`, and similar ops) produced by the compiler stage and executed sequentially by the pretend VM. Serialized IR often appears as `pretend-linked.ll` under the sandbox directory. For opcode details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Agent orchestration

`pretend-run` loads settings and a single chat model, then runs a compiled LangGraph pipeline ([`pretend_compiler/graph.py`](pretend_compiler/graph.py)). **pretend-precompiler** and **pretend-compiler** are the two LLM agents (`run_precompile`, `run_compile`); **vm** is deterministic (`execute_ir`). With `--use-files`, **write_plan** and **write_link** persist `pretend-linked.plan` and `pretend-linked.ll` under `--sandbox-dir` before the VM loads IR. **abort** aggregates validation or compiler diagnostics; **dry_run** prints plan + IR without executing the VM. A recoverable IR **fault** can loop **retry_bump → compile** until `max_retries` is reached.


For IR op types, LLM backends, and sandbox rules, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

For a command-level walkthrough of running the sample program through the pipeline, see [`docs/HelloWorld.md`](docs/HelloWorld.md).

## TL;DR

From the repo root:

```bash
echo "install deps"
uv sync
echo "pull down mistral as a local llm (mistral or qwen)"
uv run pretend-models pull mistral
echo "compile/run sample.c"
uv run pretend-run --lang c sample.c
```

The pull step downloads a large GGUF once and **writes `MISTRAL_GGUF_PATH` (or `QWEN_GGUF_PATH` for Qwen) into `./llm.env`** in your current directory, creating or updating that file automatically.

**Troubleshooting local runs:**

- **`llama_context` / `n_ctx`**, **no stdout**, **`LLM_MAX_TOKENS`**: Ensure `llm.env` doesn’t shrink limits. Defaults **`N_CTX=32768`** and **`LLM_MAX_TOKENS=8192`** avoid common truncation (you can set **`N_CTX=8192`** if the model fails to load on low RAM).
- **`--use-files`** writes **`pretend-linked.plan`** (after validation) and **`pretend-linked.ll`** (after compile) so less text sits in graph state; it does **not** fix “empty IR” from the model.
- If the **local model returns no IR** (common on CPU / small GGUF), you’ll see a compiler diagnostic; for **simple `for` + `printf` / `print` loops** the tool may **fall back to a built-in heuristic** so **`sample.c` / `sample.py`** still print ten lines. For real lowering, prefer **`--hosted`** or a larger GGUF.

## Setup

```bash
cd gpt-pretends-itsa-compiler
uv sync
```

Optional: copy [`llm.env.example`](llm.env.example) to `llm.env` if you want commented defaults before pulling. Otherwise, `pretend-models pull …` creates or updates `llm.env` with the GGUF path for you.

## Local models (default)

Pull a pinned **Mistral** or **Qwen** GGUF (Hugging Face):

```bash
uv run pretend-models pull mistral
# or
uv run pretend-models pull qwen
```

After a successful pull, `llm.env` is updated with the GGUF path. You can still edit `llm.env` by hand or copy from [`llm.env.example`](llm.env.example) for other settings.

## Sample programs (hello world × 10)

Small loops live in the repo root: [`sample.py`](sample.py), [`sample.c`](sample.c), [`sample.java`](sample.java).

### Run with a normal toolchain

From the project directory:

**Python**

```bash
python sample.py
# or
uv run python sample.py
```

**C** (needs a C compiler such as `gcc`)

```bash
gcc sample.c -o sample_c && ./sample_c
```

**Java** (needs `javac` / `java`; the class name is `HelloTen`)

```bash
javac sample.java
java HelloTen
```

### Run with the pretend compiler

Requires a local GGUF or `--hosted` with API keys (see above). Examples:

```bash
uv run pretend-run --lang python sample.py --dry-run
uv run pretend-run --lang c sample.c --dry-run
uv run pretend-run --lang java sample.java --dry-run
```

Omit `--dry-run` to execute the lowered IR in the pretend VM once models are configured.

**Low RAM / CPU-only**: pass **`--use-files`** so the precompiler plan is saved to **`pretend-linked.plan`**, then IR to **`pretend-linked.ll`** under **`--sandbox-dir`** (default: current directory), the in-memory `ir_ops` list is cleared, and the VM loads ops from the `.ll` file.

```bash
uv run pretend-run --lang c sample.c --use-files --sandbox-dir .
```

## Run

```bash
uv run pretend-run --lang c path/to/file.c
```

- **Backends**: default `mistral` (GGUF), or `--backend qwen`, `--backend ollama` (optional: `uv sync --extra ollama`).
- **Hosted API**: `uv run pretend-run --hosted --lang python app.py` (set `OPENAI_*` in `llm.env`).
- **Sandbox**: `--sandbox-dir DIR` (default: cwd). Network tools require `--allow-network`.
- **File-backed IR**: `--use-files` writes **`pretend-linked.plan`** (precompiler plan for the compiler) after validation, then **`pretend-linked.ll`** (JSON IR) after compile; VM reads the `.ll` file from the sandbox path above.

## Tests

```bash
uv run pytest
```

On low-resource machines or in CI, prefer excluding local GGUF subprocess tests (they can load large models and run for a long time) and optional hosted-API smoke tests:

```bash
uv run pytest -m "not local_llm and not hosted"
```

Integration tests that need a GGUF are marked `@pytest.mark.local_llm` and skip if no model. Optional **hosted** API smoke tests use `@pytest.mark.hosted` and run only when `OPENAI_API_KEY` is set (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
