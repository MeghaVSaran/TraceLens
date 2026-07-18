"""Tests for runtime sanitizer evidence metrics."""

import json

from src.evaluation.runtime_metrics import evaluate_runtime_cases, load_runtime_cases


UAF_LOG = """
==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1
#0 0x401111 in BufferOwner::read() /repo/fixture.cpp:15:10
freed by thread T0 here:
#0 0x7ffff in operator delete[](void*) ../../libsanitizer/asan/asan_new_delete.cpp:155
#1 0x401100 in BufferOwner::clear() /repo/fixture.cpp:10:5
previously allocated by thread T0 here:
#0 0x7fffe in operator new[](unsigned long) ../../libsanitizer/asan/asan_new_delete.cpp:98
#1 0x401080 in BufferOwner::BufferOwner() /repo/fixture.cpp:6:5
"""

UBSAN_LOG = """
/repo/overflow.cpp:8:18: runtime error: signed integer overflow
#0 0x401200 in checkedAdd(int) /repo/overflow.cpp:8:18
"""


def test_runtime_metrics_measure_evidence_and_clean_false_positives():
    report = evaluate_runtime_cases(
        [
            {
                "id": "uaf",
                "log": UAF_LOG,
                "expected": {
                    "crash_type": "heap-use-after-free",
                    "crash_functions": ["BufferOwner::read"],
                    "crash_files": ["fixture.cpp"],
                    "free_functions": ["BufferOwner::clear"],
                    "allocation_functions": ["BufferOwner::BufferOwner"],
                    "suspicious_functions": ["BufferOwner::clear"],
                },
            },
            {
                "id": "ubsan",
                "log": UBSAN_LOG,
                "expected": {
                    "crash_type": "ubsan_error",
                    "crash_functions": ["checkedAdd"],
                    "crash_files": ["overflow.cpp"],
                },
            },
            {
                "id": "clean",
                "log": "program completed successfully\n",
                "expected": {"crash_type": "unknown"},
            },
        ]
    )

    summary = report["summary"]
    assert summary["evaluated_cases"] == 3
    assert summary["crash_type_accuracy"] == 1.0
    assert summary["crash_frame_recall"] == 1.0
    assert summary["crash_file_recall"] == 1.0
    assert summary["free_stack_recall"] == 1.0
    assert summary["allocation_stack_recall"] == 1.0
    assert summary["suspicious_frame_recall"] == 1.0
    assert summary["clean_false_positive_rate"] == 0.0
    assert summary["complete_case_accuracy"] == 1.0


def test_runtime_metrics_report_exclusions_without_inflating_accuracy():
    report = evaluate_runtime_cases(
        [
            {"id": "missing-log", "expected": {"crash_type": "segfault"}},
            {"id": "missing-label", "log": "SIGSEGV"},
        ]
    )

    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["evaluated_cases"] == 0
    assert report["summary"]["excluded_cases"] == 2
    assert report["summary"]["crash_type_accuracy"] is None


def test_load_runtime_cases_supports_relative_log_files(tmp_path):
    log_path = tmp_path / "uaf.log"
    log_path.write_text(UAF_LOG, encoding="utf-8")
    dataset_path = tmp_path / "runtime.json"
    dataset_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "uaf",
                        "log_path": "uaf.log",
                        "expected": {"crash_type": "heap-use-after-free"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_runtime_cases(dataset_path)

    assert cases[0]["log"] == UAF_LOG
