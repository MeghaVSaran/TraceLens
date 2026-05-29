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
