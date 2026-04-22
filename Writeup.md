# Exploring LLMs as Compiler and Virtual Machine

This project explores a serious question: can a large language model act as a useful compiler component, and can a constrained runtime make that output executable in a controlled way?

`gpt-pretends-itsa-compiler` is an experiment built around that question. It takes source code in a declared language, runs it through LLM-driven stages that validate and lower the program, and then executes the resulting intermediate representation (IR) in a Python virtual machine. The goal is not to replace traditional compilers. The goal is to study how far model knowledge and structured orchestration can go when they are embedded inside a disciplined software system.

## Background

Large language models have seen enormous amounts of source code, programming language documentation, examples, and runtime conventions. That broad prior knowledge makes them unusually good at tasks that look compiler-adjacent:

- recognizing the language a source file appears to be written in
- identifying common syntax or structural mistakes
- summarizing program intent
- lowering high-level code into a more explicit operational form
- reasoning about how a piece of code should behave when executed

This project treats those capabilities as something worth testing directly. Instead of using an LLM only as a coding assistant, it asks whether the model can participate as part of a compiler pipeline and whether its output can drive a VM-like execution model.

The result is a system that is intentionally compiler-shaped:

1. validate the input program
2. generate a plan for lowering it
3. compile it into a constrained IR
4. execute that IR in a runtime with explicit rules

## Project goal

The main objective is to explore the boundary between model knowledge and deterministic infrastructure.

The LLM is responsible for the parts of the pipeline where broad learned knowledge is useful: language recognition, planning, and lowering. The runtime is responsible for the parts where predictable behavior matters: normalization, tool restrictions, sandboxing, execution order, output capture, and exit handling.

That split is central to the project. It makes the repository a systems experiment rather than a prompt demo.

## How the system works

The command-line entrypoint is `pretend-run`. A typical invocation looks like this:

```bash
uv run pretend-run --lang c sample.c
```

At a high level, the pipeline works as follows:

1. the CLI reads the source file and builds the configured model backend
2. the precompiler checks whether the source plausibly matches the declared language and produces a plan
3. the compiler lowers the source into a JSON IR
4. the VM interprets that IR and produces stdout, stderr, and an exit code

The project uses LangGraph to orchestrate those stages. That orchestration matters because it makes the workflow explicit, retryable, and inspectable. This is not a single prompt that directly emits output. It is a staged pipeline with state passed from one phase to the next.

## The role of the LLM

The system has two LLM-driven stages.

The **precompiler** acts as a validation and planning phase. It checks whether the source text plausibly matches the declared language and produces a structured plan describing how the program should be lowered.

The **compiler** takes that plan and emits a list of IR operations. Those operations represent a simplified execution model rather than native machine code or bytecode. The compiler stage is therefore best understood as a structured translation step.

This is where the broader knowledge of an LLM becomes relevant. A model has seen enough code patterns across languages to infer simple behavior, recognize loops and I/O, and map high-level program structure into lower-level actions. The repository is investigating how reliable that can be when the output format is constrained and validated.

## The intermediate representation

The IR is a JSON array of operation objects. It is the central contract between the model-driven compiler stage and the deterministic runtime.

Supported operations include:

- `print`
- `emit_stderr`
- `tool`
- `exit`
- `fault`

This design keeps execution narrow and explicit. Rather than allowing arbitrary generated code to run, the system only allows behavior that can be expressed through the supported operations and VM tools.

The IR can also be written to disk as `pretend-linked.ll`. When `--use-files` is enabled, the plan and IR are persisted as pipeline artifacts. That makes the lowering step easier to inspect and reduces how much large structured text needs to remain in memory.

## The virtual machine

The VM is intentionally deterministic.

Once the LLM has produced IR, execution is handled entirely by Python code in the runtime. The VM walks the operations in order, accumulates stdout and stderr, tracks exit state, and optionally dispatches tool calls through a restricted registry.

That registry includes capabilities such as:

- reading and writing files within a sandbox directory
- listing directories and checking path existence
- reading environment variables
- accessing stdin
- generating time or random values
- making HTTP requests when network access is explicitly enabled

This approach is important for two reasons.

First, it gives the system a clear execution boundary. The model does not execute code directly; it proposes a structured program for a constrained runtime to interpret.

Second, it makes behavior testable. The VM can be unit-tested independently of any model backend, which is necessary if this kind of system is going to be analyzed seriously.

## Safety and control

Although this repository is exploratory, it is not casual about execution.

File operations are resolved relative to a configured sandbox directory. Path escapes are rejected. Network access is disabled unless `--allow-network` is set. Tool calls are restricted to known names. IR is normalized before execution so malformed or low-quality model output is less likely to create confusing behavior.

These controls do not make the project equivalent to a hardened operating-system sandbox, but they do show the right architectural instinct: model output should pass through deterministic checks and constrained execution paths before it is trusted to do anything observable.

## Handling model failure

One of the more valuable parts of the project is that it treats model failure as a normal engineering concern.

If the compiler claims success but produces empty IR, that is treated as an error. If the VM encounters a recoverable fault, the graph can retry compilation with fault context. For trivial examples, the codebase also includes a heuristic fallback for simple print-loop patterns when a local model fails to produce usable IR.

That is important because any serious attempt to use LLMs in compiler-like roles has to account for inconsistency, truncation, and malformed structured output. This repository does not avoid that reality. It models it directly.

## Why this matters

The project is useful beyond its immediate implementation because it demonstrates a broader pattern for AI systems:

**use an LLM where broad learned knowledge helps, and use deterministic software everywhere that precision, control, and auditability matter.**

Compiler and VM terminology make that pattern especially concrete. A compiler has to translate intent into structure. A runtime has to execute that structure according to rules. Those are exactly the kinds of boundaries that help turn model behavior into something inspectable.

This is why the project is serious. It is not making the claim that an LLM is a drop-in replacement for a real compiler or a real VM. It is testing whether the knowledge encoded in a model can be harnessed in compiler-like and runtime-like roles when the surrounding system is designed carefully enough.

## Current limits

The project is explicit about its limits.

It is not a traditional compiler toolchain. It does not emit machine code. It does not run arbitrary source programs directly. Validation is heuristic, compilation is heuristic, and runtime behavior is limited to a small approved instruction set.

Those limits are not weaknesses in the framing of the project. They are part of the research value. They make it possible to study which parts of compilation and execution can be delegated to model-driven translation and which parts still need conventional systems design.

## Conclusion

`gpt-pretends-itsa-compiler` is best understood as an exploration of LLMs as structured systems components. It uses compiler and VM abstractions to test a practical idea: models know a great deal about programming languages and program behavior, but that knowledge becomes useful only when it is connected to deterministic validation, normalization, and execution.

The repository is therefore not just about generating code-like output. It is about building a controlled environment in which an LLM can serve as a compiler-like translator and a runtime-adjacent reasoning component while the rest of the system enforces the rules.

That makes the project a meaningful experiment in AI-assisted systems design, not a novelty.
