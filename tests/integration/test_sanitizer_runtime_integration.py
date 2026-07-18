"""Linux integration benchmark using real compiled ASan/UBSan reports."""

from pathlib import Path
import json
import os
import platform
import shutil
import subprocess

import pytest

from src.evaluation.runtime_metrics import evaluate_runtime_cases


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or not shutil.which("g++"),
    reason="requires Linux and g++ with sanitizer runtimes",
)


UAF_SOURCE = r"""
class BufferOwner {
public:
    __attribute__((noinline)) BufferOwner() : data_(new int[4]{1, 2, 3, 4}) {}
    __attribute__((noinline)) void clear() { delete[] data_; }
    __attribute__((noinline)) int read() const { return data_[0]; }
private:
    int* data_;
};

int main() {
    BufferOwner owner;
    owner.clear();
    return owner.read();
}
"""

HEAP_OVERFLOW_SOURCE = r"""
__attribute__((noinline)) void writePastEnd(int* data) {
    data[4] = 99;
}

int main() {
    int* data = new int[4]{};
    writePastEnd(data);
    delete[] data;
    return 0;
}
"""

UBSAN_SOURCE = r"""
#include <climits>

__attribute__((noinline)) int checkedAdd(int value) {
    return value + 1;
}

int main() {
    volatile int value = INT_MAX;
    return checkedAdd(value);
}
"""

LEAK_SOURCE = r"""
__attribute__((noinline)) int* allocateLeak() {
    return new int[4]{1, 2, 3, 4};
}

int main() {
    volatile int* leaked = allocateLeak();
    volatile int value = leaked[0];
    (void)value;
    return 0;
}
"""

CLEAN_SOURCE = r"""
#include <iostream>

int main() {
    int values[4] = {1, 2, 3, 4};
    std::cout << values[2] << "\n";
    return 0;
}
"""


def test_real_sanitizer_runtime_evaluation(tmp_path: Path):
    uaf = _compile_and_run(
        tmp_path,
        "uaf_fixture",
        UAF_SOURCE,
        ["-fsanitize=address"],
        {"ASAN_OPTIONS": "abort_on_error=1:detect_leaks=0:symbolize=1"},
        expect_failure=True,
    )
    overflow = _compile_and_run(
        tmp_path,
        "heap_overflow_fixture",
        HEAP_OVERFLOW_SOURCE,
        ["-fsanitize=address"],
        {"ASAN_OPTIONS": "abort_on_error=1:detect_leaks=0:symbolize=1"},
        expect_failure=True,
    )
    ubsan = _compile_and_run(
        tmp_path,
        "ubsan_fixture",
        UBSAN_SOURCE,
        ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"],
        {"UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"},
        expect_failure=True,
    )
    leak = _compile_and_run(
        tmp_path,
        "leak_fixture",
        LEAK_SOURCE,
        ["-fsanitize=address"],
        {
            "ASAN_OPTIONS": (
                "detect_leaks=1:halt_on_error=1:exitcode=23:symbolize=1"
            )
        },
        expect_failure=True,
    )
    clean = _compile_and_run(
        tmp_path,
        "clean_fixture",
        CLEAN_SOURCE,
        ["-fsanitize=address,undefined"],
        {
            "ASAN_OPTIONS": "abort_on_error=1:detect_leaks=1:symbolize=1",
            "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1",
        },
        expect_failure=False,
    )

    report = evaluate_runtime_cases(
        [
            {
                "id": "compiled-uaf",
                "log": uaf,
                "expected": {
                    "crash_type": "heap-use-after-free",
                    "crash_functions": ["BufferOwner::read"],
                    "crash_files": ["uaf_fixture.cpp"],
                    "free_functions": ["BufferOwner::clear"],
                    "allocation_functions": ["BufferOwner::BufferOwner"],
                    "suspicious_functions": ["BufferOwner::clear"],
                },
            },
            {
                "id": "compiled-heap-overflow",
                "log": overflow,
                "expected": {
                    "crash_type": "heap-buffer-overflow",
                    "crash_functions": ["writePastEnd"],
                    "crash_files": ["heap_overflow_fixture.cpp"],
                },
            },
            {
                "id": "compiled-ubsan",
                "log": ubsan,
                "expected": {
                    "crash_type": "ubsan_error",
                    "crash_functions": ["checkedAdd"],
                    "crash_files": ["ubsan_fixture.cpp"],
                },
            },
            {
                "id": "compiled-memory-leak",
                "log": leak,
                "expected": {
                    "crash_type": "memory_leak",
                    "allocation_functions": ["allocateLeak"],
                },
            },
            {
                "id": "compiled-clean",
                "log": clean,
                "expected": {"crash_type": "unknown"},
            },
        ],
        repo_root=tmp_path,
    )

    print(json.dumps(report["summary"], indent=2))
    assert report["summary"]["evaluated_cases"] == 5
    assert report["summary"]["crash_type_accuracy"] == 1.0
    assert report["summary"]["positive_parse_coverage"] == 1.0
    assert report["summary"]["crash_frame_recall"] == 1.0
    assert report["summary"]["crash_file_recall"] == 1.0
    assert report["summary"]["free_stack_recall"] == 1.0
    assert report["summary"]["allocation_stack_recall"] == 1.0
    assert report["summary"]["suspicious_frame_recall"] == 1.0
    assert report["summary"]["clean_false_positive_rate"] == 0.0
    assert report["summary"]["complete_case_accuracy"] == 1.0


def _compile_and_run(
    tmp_path: Path,
    name: str,
    source_text: str,
    sanitizer_flags: list[str],
    environment: dict[str, str],
    *,
    expect_failure: bool,
) -> str:
    source = tmp_path / f"{name}.cpp"
    binary = tmp_path / name
    source.write_text(source_text, encoding="utf-8")

    compile_command = [
        shutil.which("g++"),
        "-std=c++17",
        "-g",
        "-O0",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
        "-no-pie",
        "-rdynamic",
        *sanitizer_flags,
        str(source),
        "-o",
        str(binary),
    ]
    compiled = subprocess.run(
        compile_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    process_env = os.environ.copy()
    process_env.update(environment)
    completed = subprocess.run(
        [str(binary)],
        check=False,
        capture_output=True,
        text=True,
        env=process_env,
        timeout=30,
    )
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    if expect_failure:
        assert completed.returncode != 0, output
        assert "Sanitizer" in output or "runtime error:" in output
    else:
        assert completed.returncode == 0, output
    return output or "program completed successfully"
