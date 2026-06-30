"""Crash-report triage built on parsed runtime logs and retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Optional

from src.retrieval.hybrid_retriever import RetrievalResult


_FRAME_RE = re.compile(
    r"^#\s*(?P<index>\d+)\s+"
    r"(?:0x[0-9a-fA-F]+\s+)?"
    r"(?:in\s+)?(?P<function>[A-Za-z_~][A-Za-z0-9_:<>~*]*)"
    r".*?(?:\s+at\s+|\s+)(?P<file>[\w./\\-]+\.(?:cc|cpp|cxx|c|h|hpp|hxx))"
    r"(?::(?P<line>\d+))?",
)


@dataclass
class CrashFrame:
    """Structured stack-frame evidence extracted from a crash report."""

    index: int
    function: str
    file_path: str = ""
    line_number: int = 0
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrashTriageReport:
    """Actionable crash diagnosis assembled from logs, search, and retrieval."""

    crash_type: str
    summary: str
    likely_cause: str
    suggested_next_steps: list[str] = field(default_factory=list)
    crash_frame: Optional[CrashFrame] = None
    suspicious_frames: list[CrashFrame] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    exact_matches: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    confidence: str = "low"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["crash_frame"] = self.crash_frame.to_dict() if self.crash_frame else None
        data["suspicious_frames"] = [frame.to_dict() for frame in self.suspicious_frames]
        return data


def analyze_crash(
    parsed_log,
    results: Iterable[RetrievalResult],
    repo_root: Optional[Path] = None,
) -> CrashTriageReport:
    """Create a deterministic first-pass crash report.

    The analyzer intentionally does not pretend to prove the bug. It combines
    sanitizer/GDB signals, exact symbol search, and top retrieval evidence so a
    developer can jump to the highest-value frames quickly.
    """
    results = list(results)
    raw_log = getattr(parsed_log, "raw_log", "") or ""
    crash_type = _classify_crash(raw_log, getattr(parsed_log, "error_type", "unknown"))
    frames = _parse_frames(list(getattr(parsed_log, "stack_frames", []) or []))
    crash_frame = frames[0] if frames else None
    suspicious_frames = _choose_suspicious_frames(frames, results)
    exact_matches = _exact_symbol_matches(
        repo_root=repo_root,
        identifiers=list(getattr(parsed_log, "identifiers", []) or []),
    )
    relevant_files = _top_files(results, frames)
    evidence = _build_evidence(parsed_log, results, frames, exact_matches)
    summary = _summary(crash_type, crash_frame, results)
    likely_cause = _likely_cause(crash_type, crash_frame, suspicious_frames, results)
    confidence = _confidence(crash_type, crash_frame, results, exact_matches)

    return CrashTriageReport(
        crash_type=crash_type,
        summary=summary,
        likely_cause=likely_cause,
        suggested_next_steps=_suggested_steps(crash_type),
        crash_frame=crash_frame,
        suspicious_frames=suspicious_frames,
        evidence=evidence,
        exact_matches=exact_matches,
        relevant_files=relevant_files,
        confidence=confidence,
    )


def _classify_crash(raw_log: str, error_type: str) -> str:
    text = raw_log.lower()
    if "heap-use-after-free" in text or "use-after-free" in text:
        return "heap-use-after-free"
    if "heap-buffer-overflow" in text:
        return "heap-buffer-overflow"
    if "stack-buffer-overflow" in text:
        return "stack-buffer-overflow"
    if "addresssanitizer" in text:
        return "asan_error"
    if "undefinedbehaviorsanitizer" in text or "runtime error:" in text:
        return "ubsan_error"
    if "leaksanitizer" in text or "detected memory leaks" in text:
        return "memory_leak"
    if "segmentation fault" in text or "sigsegv" in text:
        return "segfault"
    return error_type or "unknown"


def _parse_frames(raw_frames: list[str]) -> list[CrashFrame]:
    frames: list[CrashFrame] = []
    for raw in raw_frames:
        match = _FRAME_RE.search(raw.strip())
        if not match:
            continue
        frames.append(
            CrashFrame(
                index=int(match.group("index")),
                function=match.group("function"),
                file_path=(match.group("file") or "").replace("\\", "/"),
                line_number=int(match.group("line") or 0),
                raw=raw.strip(),
            )
        )
    return frames


def _choose_suspicious_frames(
    frames: list[CrashFrame],
    results: list[RetrievalResult],
    limit: int = 3,
) -> list[CrashFrame]:
    if not frames:
        return []
    result_files = {result.file_path.replace("\\", "/") for result in results}
    chosen: list[CrashFrame] = []
    for frame in frames:
        if frame.file_path and (
            frame.file_path in result_files
            or any(path.endswith(frame.file_path) for path in result_files)
        ):
            chosen.append(frame)
        if len(chosen) >= limit:
            return chosen
    return frames[:limit]


def _exact_symbol_matches(repo_root: Optional[Path], identifiers: list[str]) -> list[str]:
    if repo_root is None or not identifiers:
        return []
    repo_root = Path(repo_root).resolve()
    matches: list[str] = []
    for symbol in identifiers[:5]:
        matches.extend(_search_symbol(repo_root, symbol, max_matches=3))
        if len(matches) >= 8:
            break
    return matches[:8]


def _search_symbol(repo_root: Path, symbol: str, max_matches: int = 3) -> list[str]:
    if not symbol:
        return []
    rg = shutil.which("rg")
    if rg:
        try:
            completed = subprocess.run(
                [
                    rg,
                    "--line-number",
                    "--fixed-strings",
                    "--glob",
                    "*.cc",
                    "--glob",
                    "*.cpp",
                    "--glob",
                    "*.cxx",
                    "--glob",
                    "*.h",
                    "--glob",
                    "*.hpp",
                    "--glob",
                    "*.hxx",
                    symbol,
                    str(repo_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            return _normalize_rg_lines(completed.stdout, repo_root, max_matches=max_matches)
        except (OSError, subprocess.SubprocessError):
            pass
    return _fallback_search(repo_root, symbol, max_matches=max_matches)


def _normalize_rg_lines(stdout: str, repo_root: Path, max_matches: int) -> list[str]:
    lines: list[str] = []
    for raw in stdout.splitlines():
        match = re.match(r"^(?P<path>.+):(?P<line>\d+):", raw)
        if not match:
            continue
        path = Path(match.group("path"))
        try:
            display_path = path.resolve().relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            display_path = path.as_posix()
        line = match.group("line")
        lines.append(f"{display_path}:{line}")
        if len(lines) >= max_matches:
            break
    return lines


def _fallback_search(repo_root: Path, symbol: str, max_matches: int) -> list[str]:
    suffixes = {".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".hxx"}
    matches: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if symbol in line:
                    matches.append(f"{path.relative_to(repo_root).as_posix()}:{number}")
                    if len(matches) >= max_matches:
                        return matches
        except OSError:
            continue
    return matches


def _top_files(results: list[RetrievalResult], frames: list[CrashFrame]) -> list[str]:
    files: list[str] = []
    for frame in frames:
        if frame.file_path and frame.file_path not in files:
            files.append(frame.file_path)
    for result in results:
        if result.file_path not in files:
            files.append(result.file_path)
        if len(files) >= 5:
            break
    return files[:5]


def _build_evidence(parsed_log, results, frames, exact_matches) -> list[str]:
    evidence: list[str] = []
    identifiers = list(getattr(parsed_log, "identifiers", []) or [])
    if identifiers:
        evidence.append("Runtime symbols extracted: " + ", ".join(identifiers[:5]))
    if frames:
        evidence.append(f"Stack frames parsed: {len(frames)}")
        evidence.append(f"Crash frame: #{frames[0].index} {frames[0].function}")
    if results:
        top = results[0]
        evidence.append(
            f"Top retrieval result: {top.file_path}:{top.start_line} ({top.function_name}) score={top.score:.3f}"
        )
    if exact_matches:
        evidence.append(f"Exact symbol search found {len(exact_matches)} repo location(s).")
    return evidence


def _summary(crash_type: str, crash_frame: Optional[CrashFrame], results: list[RetrievalResult]) -> str:
    if crash_frame:
        return f"{crash_type} triaged around frame #{crash_frame.index} {crash_frame.function}."
    if results:
        top = results[0]
        return f"{crash_type} triaged from retrieval evidence near {top.file_path}:{top.start_line}."
    return f"{crash_type} detected, but no precise code frame was recovered."


def _likely_cause(
    crash_type: str,
    crash_frame: Optional[CrashFrame],
    suspicious_frames: list[CrashFrame],
    results: list[RetrievalResult],
) -> str:
    anchor = ""
    if suspicious_frames:
        anchor = f" Suspicious frame to inspect first: #{suspicious_frames[0].index} {suspicious_frames[0].function}."
    elif crash_frame:
        anchor = f" Start at crash frame #{crash_frame.index} {crash_frame.function}."
    elif results:
        anchor = f" Start with retrieved context {results[0].file_path}:{results[0].start_line}."

    if crash_type == "heap-use-after-free":
        return "Memory is likely accessed after its lifetime ended." + anchor
    if crash_type in {"heap-buffer-overflow", "stack-buffer-overflow"}:
        return "Code likely reads/writes outside a valid buffer range." + anchor
    if crash_type == "memory_leak":
        return "Allocated memory is likely not released on at least one path." + anchor
    if crash_type == "ubsan_error":
        return "Undefined behavior was reported; inspect the operation and its caller-provided values." + anchor
    if crash_type == "segfault":
        return "A null, dangling, or invalid pointer/reference is likely being dereferenced." + anchor
    return "Runtime failure needs inspection of the top crash/retrieval evidence." + anchor


def _suggested_steps(crash_type: str) -> list[str]:
    if crash_type == "heap-use-after-free":
        return [
            "Inspect the free/delete/reset site and the later access site in stack order.",
            "Check ownership transfer and whether raw pointers outlive their owner.",
            "Consider RAII ownership, invalidation after free, or copying needed state before release.",
        ]
    if crash_type in {"heap-buffer-overflow", "stack-buffer-overflow"}:
        return [
            "Check index arithmetic and loop bounds at the crash frame.",
            "Compare requested access size with container/allocation size.",
            "Inspect caller-provided length/capacity values one or two frames above the crash.",
        ]
    if crash_type == "memory_leak":
        return [
            "Find the allocation path and the normal/error cleanup paths.",
            "Check early returns and exception paths for missing release.",
            "Prefer RAII containers or smart pointers where ownership is local.",
        ]
    return [
        "Inspect the crash frame, then move upward through callers until the bad state is introduced.",
        "Use exact symbol matches and top retrieval results as jump points.",
        "Rerun with ASan/UBSan/GDB backtrace if this log lacks stack detail.",
    ]


def _confidence(crash_type: str, crash_frame, results, exact_matches) -> str:
    if crash_type in {"heap-use-after-free", "heap-buffer-overflow", "stack-buffer-overflow"} and crash_frame and results:
        return "high"
    if crash_frame and (results or exact_matches):
        return "medium"
    if results or exact_matches:
        return "medium"
    return "low"

