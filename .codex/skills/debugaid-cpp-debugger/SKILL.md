---
name: debugaid-cpp-debugger
description: Develop and validate DebugAid, an evidence-first C++ failure-triage and debugging assistant. Use for changes involving C++ build failures, GDB/core dumps, ASan/UBSan reports, memory or runtime diagnostics, repository search, symbol/build reasoning, RAG, agentic investigation, evaluation datasets, CLI workflows, or debugging-related architecture in the log2code repository.
---

# DebugAid C++ Debugger

Build DebugAid as a local, repo-aware assistant that orchestrates proven C++ tools and explains their evidence. Treat an LLM as an investigation planner and synthesis layer, never as the source of runtime or static-analysis facts.

## Workflow

1. Read `docs/8_project_vision_and_complete_project_specification` and inspect the relevant implementation and tests.
2. Run `git status --short --branch`. Preserve all existing changes and avoid broad cleanup.
3. State one measurable capability for the change. Prefer one analyzer, collector, or evaluation improvement per iteration.
4. Establish the deterministic evidence path before adding retrieval or AI:
   - Runtime state: GDB, core dumps, ASan, UBSan, LSan.
   - Symbols: `addr2line`, `nm`, `c++filt`, compile commands, AST/symbol indexes.
   - Source: `rg`, then `git grep`, then bounded Python filesystem search.
   - Build: `compile_commands.json`, CMake targets, source ownership, link edges.
   - Preventive findings: compiler warnings, clang-tidy, Clang Static Analyzer, or cppcheck.
5. Keep observations, inferences, and recommendations distinct in data structures and output.
6. Add parser/unit tests using realistic raw tool output. Mock external tools for unit tests.
7. Add an integration fixture when behavior depends on a compiler, debugger, sanitizer, binary, or core dump.
8. Run focused tests, then the full suite. Report exact results and any untested system dependency.
9. Update architecture or user documentation only when behavior changed. Do not rewrite historical agent logs.

## Guardrails

- Do not claim a root cause is proven unless direct evidence establishes it.
- Do not claim broad C++ accuracy from one repository or synthetic-only data.
- Do not train a model until a versioned dataset, split policy, baseline, and leakage checks exist.
- Do not make the LLM generate uncited file paths, symbols, target relationships, or analyzer findings.
- Assign confidence from explicit evidence rules and expose the supporting evidence.
- Use subprocess argument lists with `shell=False`, timeouts, bounded output, and captured tool versions.
- Treat logs, source, debugger memory, and model prompts as potentially sensitive.
- Keep the deterministic product useful when embeddings, a GPU, or an LLM are unavailable.

## Environment Routing

- Use Windows for Python implementation, mocked tool tests, CLI tests, retrieval, and documentation.
- Request WSL/Linux only for real GCC/Clang, GDB, sanitizer, core-dump, `perf`, or native-tool integration tests.
- Request Colab/GPU only for large embedding benchmarks, reranker training, or model evaluation.
- When user action is required, provide one copy-paste setup block, one execution block, expected artifacts, and the exact files or output to return.

## Evaluation

Read [evaluation.md](references/evaluation.md) before adding datasets, claiming accuracy, training a model, or declaring a debugging capability complete.

For each capability, measure correctness and usefulness separately. At minimum record parser success, evidence localization, false positives, latency, and failure behavior when tools or symbols are unavailable.

## Product Sequence

Prefer this order unless current evidence demonstrates a different bottleneck:

1. Reliable log/sanitizer/core-dump evidence collection.
2. Symbolization and exact repository/build correlation.
3. Failure-specific deterministic diagnosis.
4. Real multi-repository evaluation.
5. Bounded agentic investigation over structured tools.
6. Optional cited LLM explanation.
7. Learned reranking only after the dataset supports it.
