# DebugAid

**ML-Assisted Log–Code Correlation Engine**

Maps C++ build and runtime error logs to the most relevant source files and
functions using semantic embeddings, hybrid retrieval, and early Phase 2
build-aware triage.

## What It Does

```
$ debugaid query --log build.log --repo ./llvm-project

Analyzing log: build.log
Error type: linker_error
Query: "undefined reference to Parser::resolveSymbol"

Top matches:

1. src/parser/resolve.cpp         Parser::resolveSymbol()    [line 142]  score: 0.91
2. include/parser/symbol_table.h  SymbolTable::lookup()      [line 67]   score: 0.84
3. src/compiler/analyze.cpp       Compiler::analyze()        [line 203]  score: 0.79
4. src/parser/parse_expr.cpp      Parser::parseExpression()  [line 88]   score: 0.71
5. src/linker/link_resolver.cpp   Linker::resolveAll()       [line 315]  score: 0.68
```

## How It Works

1. **Code Ingestion** — Tree-sitter parses C++ repository into function-level chunks
2. **Code Embedding** — GraphCodeBERT converts each function into a 768-dim vector
3. **Log Parsing** — Regex extracts error type, symbol names, and file hints
4. **Log Embedding** — all-mpnet-base-v2 converts log into a 768-dim vector
5. **Hybrid Retrieval** — Dense (ChromaDB) + Sparse (BM25) search, score fusion
6. **Diagnosis Layer** — Optional compile_commands-aware findings for include/link/build failures
7. **Ranked Output** — Top-K files and functions returned with scores

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Index a repository (run once)
debugaid index --repo ./llvm-project

# Query with a log file
debugaid query --log build.log --repo ./llvm-project

# Query with diagnosis if compile_commands.json is available
debugaid query --log build.log --repo ./llvm-project --diagnose --build-dir ./build

# Wrap a failing build/test command and auto-triage it
debugaid watch --repo ./llvm-project --build-dir ./build -- make -j8

# Evaluate on ground truth dataset
debugaid eval --dataset data/ground_truth/dev.json --repo ./llvm-project

# Show index info
debugaid info --repo ./llvm-project
```

## Evaluation Results

| Metric    | Score |
|-----------|-------|
| Recall@1  | —     |
| Recall@3  | —     |
| Recall@5  | —     |
| MRR       | —     |

*(Filled in after evaluation run)*

## Supported Error Types

- Linker errors (`undefined reference`, `multiple definition`)
- Compiler errors (`undeclared identifier`, `no matching function`)
- Include errors (`no such file or directory`)
- Template errors (`implicit instantiation`)
- Segfault stack traces
- Build-system errors / compile database aware triage

## Project Structure

```
src/
  ingestion/      ← code_parser.py, log_parser.py
  embeddings/     ← code_embedder.py, log_embedder.py
  indexing/       ← vector_index.py, bm25_index.py
  retrieval/      ← hybrid_retriever.py
  evaluation/     ← metrics.py, benchmark.py
  cli/            ← main.py
scripts/          ← dataset mining and generation
tests/            ← pytest test suite
docs/             ← architecture, spec, roadmap
```

## Architecture

See [docs/2_system_architecture.md](docs/2_system_architecture.md)

## Roadmap

See [docs/6_roadmap.md](docs/6_roadmap.md)
