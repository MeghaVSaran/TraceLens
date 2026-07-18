# Real-World Evaluation Manifest

- Source dataset: `data/processed/github_pairs.json`
- Total usable GitHub samples: **71**

## Repository Slices

| Repo | Profile | Samples | Error Types | Dataset |
| --- | --- | ---: | --- | --- |
| `abseil/abseil-cpp` | small_modern_cpp_library | 21 | compiler_error:10, include_error:1, linker_error:9, segfault:1 | `data/real_eval/abseil_abseil_cpp.json` |
| `llvm/llvm-project` | very_large_compiler_infra | 42 | compiler_error:10, linker_error:8, segfault:24 | `data/real_eval/llvm_llvm_project.json` |
| `opencv/opencv` | large_runtime_library | 8 | compiler_error:2, segfault:6 | `data/real_eval/opencv_opencv.json` |

## Colab Evaluation Template

For each repo, clone it, build a compile database when practical, index it, then run strict-pathless ablation.

### abseil/abseil-cpp

```bash
git clone --depth=1 https://github.com/abseil/abseil-cpp.git /tmp/abseil_abseil_cpp
python -m src.cli.main index --repo /tmp/abseil_abseil_cpp --device cuda --embedding-model mpnet --include-tests --force-reindex
python -m src.evaluation.run_ablation --dataset data/real_eval/abseil_abseil_cpp.json --repo /tmp/abseil_abseil_cpp --strict-pathless --skip-missing-ground-truth
```

### llvm/llvm-project

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git /tmp/llvm_llvm_project
python -m src.cli.main index --repo /tmp/llvm_llvm_project --device cuda --embedding-model mpnet --include-tests --force-reindex
python -m src.evaluation.run_ablation --dataset data/real_eval/llvm_llvm_project.json --repo /tmp/llvm_llvm_project --strict-pathless --skip-missing-ground-truth
```

### opencv/opencv

```bash
git clone --depth=1 https://github.com/opencv/opencv.git /tmp/opencv_opencv
python -m src.cli.main index --repo /tmp/opencv_opencv --device cuda --embedding-model mpnet --include-tests --force-reindex
python -m src.evaluation.run_ablation --dataset data/real_eval/opencv_opencv.json --repo /tmp/opencv_opencv --strict-pathless --skip-missing-ground-truth
```

## Next Mining Targets

| Repo | Profile | Why |
| --- | --- | --- |
| `verilator/verilator` | open_source_eda_simulator | EDA-relevant C++ compiler/simulator with realistic build, parser, and runtime failures. |
| `YosysHQ/yosys` | open_source_synthesis_tool | EDA-relevant synthesis codebase with C++ passes and frontend/backend failures. |
| `The-OpenROAD-Project/OpenROAD` | open_source_physical_design | Closest public proxy for large EDA toolchains and complex CMake/dependency issues. |
| `protocolbuffers/protobuf` | large_codegen_runtime | Generated-code and ABI/build failures are common in industrial C++. |
| `fmtlib/fmt` | small_template_heavy_library | Small but template-heavy, useful for quick pathless compiler-error checks. |

## Training Gate

Do not train a model yet. First build a reliable multi-repo eval set with at least 300-500 high-quality real samples, including EDA-like repos. Then compare BM25, dense, hybrid, reranker, and any trained model on held-out repos.

## Multi-Repository Matrix Run

After indexing the repositories under one directory, run the same benchmark independently and keep each report separate:

```bash
python scripts/evaluate_repo_matrix.py \
  --manifest data/real_eval/manifest.json \
  --repos-root /tmp \
  --output-dir data/multi_repo_eval \
  --strict-pathless \
  --skip-missing-ground-truth
```

The matrix marks repositories as `ok`, `missing_repo`, `missing_index`, `missing_dataset`, `failed`, or `timeout`. It does not convert unavailable repositories into zero scores. Strict-pathless mode removes source paths and filenames completely before retrieval; it does not preserve basenames.

## Large-Repository Indexing

The index command now supports repository-relative path prefixes and bounded
embedding batches. Use prefixes for large monorepos so unrelated trees are not
parsed. Run one indexing command to completion before starting the matrix.

For the current 71-sample matrix on Colab, clone using the manifest slugs:

    git clone --depth=1 https://github.com/abseil/abseil-cpp.git /tmp/abseil_abseil_cpp
    git clone --depth=1 https://github.com/llvm/llvm-project.git /tmp/llvm_llvm_project
    git clone --depth=1 https://github.com/opencv/opencv.git /tmp/opencv_opencv

Start with tests excluded for LLVM; this is the reliable baseline:

    python -m src.cli.main index \
      --repo /tmp/llvm_llvm_project \
      --device cuda \
      --embedding-model mpnet \
      --embedding-batch-size 64 \
      --path-prefix clang \
      --path-prefix clang-tools-extra \
      --path-prefix compiler-rt \
      --path-prefix flang \
      --path-prefix libc \
      --path-prefix libcxx \
      --path-prefix lld \
      --path-prefix lldb \
      --path-prefix llvm \
      --path-prefix mlir \
      --force-reindex

For OpenCV, index the modules tree and keep tests only if the run completes
with sufficient memory:

    python -m src.cli.main index \
      --repo /tmp/opencv_opencv \
      --device cuda \
      --embedding-model mpnet \
      --embedding-batch-size 64 \
      --path-prefix modules \
      --include-tests \
      --force-reindex

Evaluate only repositories that were cloned and indexed:

    python scripts/evaluate_repo_matrix.py \
      --manifest data/real_eval/manifest.json \
      --repos-root /tmp \
      --output-dir data/multi_repo_eval \
      --repo-slug llvm_llvm_project \
      --repo-slug opencv_opencv \
      --strict-pathless \
      --skip-missing-ground-truth

A matrix row is incomplete_index when a previous interrupted indexing run
left .debugaid without all required artifacts. Re-run indexing with
--force-reindex rather than evaluating that row.
