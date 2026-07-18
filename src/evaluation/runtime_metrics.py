"""Honest evidence-localization metrics for runtime sanitizer triage."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Optional

from src.analysis.crash_analyzer import CrashFrame, analyze_crash
from src.ingestion.log_parser import parse_log


def load_runtime_cases(dataset_path: Path) -> list[dict]:
    """Load inline logs or dataset-relative log files from JSON."""
    dataset_path = Path(dataset_path).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Runtime dataset must be a list or an object with a 'cases' list.")

    loaded: list[dict] = []
    for raw in cases:
        case = dict(raw)
        if not case.get("log") and case.get("log_path"):
            log_path = (dataset_path.parent / case["log_path"]).resolve()
            case["log"] = log_path.read_text(encoding="utf-8", errors="replace")
        loaded.append(case)
    return loaded


def evaluate_runtime_cases(
    cases: Iterable[dict],
    *,
    repo_root: Optional[Path] = None,
) -> dict:
    """Evaluate deterministic crash evidence without retrieval or an LLM."""
    results: list[dict] = []
    exclusions: list[dict] = []

    for index, raw_case in enumerate(cases):
        case = dict(raw_case)
        case_id = str(case.get("id") or f"case-{index + 1}")
        log = case.get("log")
        expected = case.get("expected")
        if not isinstance(log, str) or not log.strip():
            exclusions.append({"id": case_id, "reason": "missing log"})
            continue
        if not isinstance(expected, dict) or not expected.get("crash_type"):
            exclusions.append({"id": case_id, "reason": "missing expected.crash_type"})
            continue

        parsed = parse_log(log, repo_root=repo_root)
        report = analyze_crash(parsed, [], repo_root=repo_root)
        results.append(_evaluate_case(case_id, expected, report))

    return {
        "summary": _aggregate(results, exclusions, total_seen=len(results) + len(exclusions)),
        "cases": results,
        "exclusions": exclusions,
    }


def _evaluate_case(case_id: str, expected: dict, report) -> dict:
    expected_type = str(expected["crash_type"])
    expected_crash_functions = _string_list(expected.get("crash_functions"))
    expected_crash_files = _string_list(expected.get("crash_files"))
    expected_free_functions = _string_list(expected.get("free_functions"))
    expected_allocation_functions = _string_list(expected.get("allocation_functions"))
    expected_suspicious_functions = _string_list(expected.get("suspicious_functions"))

    type_correct = report.crash_type == expected_type
    crash_frame_hit = _function_hit(
        [report.crash_frame] if report.crash_frame else [],
        expected_crash_functions,
    )
    crash_file_hit = _file_hit(
        [report.crash_frame] if report.crash_frame else [],
        expected_crash_files,
    )
    free_frame_hit = _function_hit(report.freed_by_frames, expected_free_functions)
    allocation_frame_hit = _function_hit(
        report.allocated_by_frames,
        expected_allocation_functions,
    )
    suspicious_frame_hit = _function_hit(
        report.suspicious_frames,
        expected_suspicious_functions,
    )
    false_positive = expected_type == "unknown" and (
        report.crash_type != "unknown"
        or report.crash_frame is not None
        or bool(report.freed_by_frames)
        or bool(report.allocated_by_frames)
    )

    checks = [
        type_correct,
        _required_hit(expected_crash_functions, crash_frame_hit),
        _required_hit(expected_crash_files, crash_file_hit),
        _required_hit(expected_free_functions, free_frame_hit),
        _required_hit(expected_allocation_functions, allocation_frame_hit),
        _required_hit(expected_suspicious_functions, suspicious_frame_hit),
        not false_positive,
    ]
    return {
        "id": case_id,
        "expected_crash_type": expected_type,
        "predicted_crash_type": report.crash_type,
        "type_correct": type_correct,
        "crash_frame_hit": crash_frame_hit,
        "crash_file_hit": crash_file_hit,
        "free_frame_hit": free_frame_hit,
        "allocation_frame_hit": allocation_frame_hit,
        "suspicious_frame_hit": suspicious_frame_hit,
        "false_positive": false_positive,
        "complete_case_correct": all(checks),
        "confidence": report.confidence,
        "predicted": {
            "crash_frame": _frame_name(report.crash_frame),
            "crash_file": report.crash_frame.file_path if report.crash_frame else "",
            "free_functions": [_frame_name(frame) for frame in report.freed_by_frames],
            "allocation_functions": [
                _frame_name(frame) for frame in report.allocated_by_frames
            ],
            "suspicious_functions": [
                _frame_name(frame) for frame in report.suspicious_frames
            ],
        },
    }


def _aggregate(results: list[dict], exclusions: list[dict], total_seen: int) -> dict:
    positive = [item for item in results if item["expected_crash_type"] != "unknown"]
    clean = [item for item in results if item["expected_crash_type"] == "unknown"]
    return {
        "total_cases": total_seen,
        "evaluated_cases": len(results),
        "excluded_cases": len(exclusions),
        "crash_type_accuracy": _rate(results, "type_correct"),
        "positive_parse_coverage": _ratio(
            sum(item["predicted_crash_type"] != "unknown" for item in positive),
            len(positive),
        ),
        "crash_frame_recall": _applicable_rate(results, "crash_frame_hit"),
        "crash_file_recall": _applicable_rate(results, "crash_file_hit"),
        "free_stack_recall": _applicable_rate(results, "free_frame_hit"),
        "allocation_stack_recall": _applicable_rate(results, "allocation_frame_hit"),
        "suspicious_frame_recall": _applicable_rate(results, "suspicious_frame_hit"),
        "clean_false_positive_rate": _ratio(
            sum(item["false_positive"] for item in clean),
            len(clean),
        ),
        "complete_case_accuracy": _rate(results, "complete_case_correct"),
        "confidence_counts": {
            level: sum(item["confidence"] == level for item in results)
            for level in ("high", "medium", "low")
        },
    }


def _applicable_rate(results: list[dict], field: str) -> Optional[float]:
    applicable = [item[field] for item in results if item[field] is not None]
    return _ratio(sum(bool(value) for value in applicable), len(applicable))


def _rate(results: list[dict], field: str) -> Optional[float]:
    return _ratio(sum(bool(item[field]) for item in results), len(results))


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _required_hit(expected: list[str], actual: Optional[bool]) -> bool:
    return True if not expected else actual is True


def _function_hit(frames: Iterable[Optional[CrashFrame]], expected: list[str]) -> Optional[bool]:
    if not expected:
        return None
    actual = {
        _normalize_function(frame.function)
        for frame in frames
        if frame is not None and frame.function
    }
    wanted = {_normalize_function(function) for function in expected}
    return bool(actual & wanted)


def _file_hit(frames: Iterable[Optional[CrashFrame]], expected: list[str]) -> Optional[bool]:
    if not expected:
        return None
    actual = [
        frame.file_path.replace("\\", "/")
        for frame in frames
        if frame is not None and frame.file_path
    ]
    return any(
        path == wanted.replace("\\", "/")
        or path.endswith("/" + wanted.replace("\\", "/").lstrip("/"))
        for path in actual
        for wanted in expected
    )


def _normalize_function(function: str) -> str:
    value = re.sub(r"\s+", " ", (function or "").strip())
    if "(" in value and not value.startswith("operator"):
        value = value.split("(", 1)[0].strip()
    return value


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def _frame_name(frame: Optional[CrashFrame]) -> str:
    return frame.function if frame else ""
