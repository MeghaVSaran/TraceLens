"""Crash-report triage built on parsed runtime logs and retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Optional

from src.retrieval.hybrid_retriever import RetrievalResult


OWNERSHIP_KEYWORDS = [
    "delete", "free", "reset", "clear", "release",
    "destroy", "remove", "erase", "pop", "move",
]
BOUNDS_KEYWORDS = [
    "size", "capacity", "index", "length", "count",
    "offset", "resize", "reserve", "push", "at",
]
CPP_SUFFIXES = {".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".hxx"}

_FRAME_PREFIX_RE = re.compile(r"^#\s*(?P<index>\d+)\s+(?P<body>.+)$")
_FRAME_FILE_RE = re.compile(
    r"(?P<file>[\w./\\-]+\.(?:cc|cpp|cxx|c|h|hpp|hxx))"
    r"(?::(?P<line>\d+))?(?::\d+)?(?:\s|$)"
)
_FRAME_START_RE = re.compile(r"^#\s*\d+\b")
_FRAME_ADDRESS_RE = re.compile(r"^(?:0x[0-9a-fA-F]+)\s+")


@dataclass
class CrashFrame:
    """Structured stack-frame evidence extracted from a crash report."""

    index: int
    function: str
    file_path: str = ""
    line_number: int = 0
    raw: str = ""
    score: float = 0.0
    evidence_source: str = "crash"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrashTriageReport:
    """Actionable crash diagnosis assembled from logs, search, and retrieval."""

    crash_type: str
    crash_frame: Optional[CrashFrame] = None
    freed_by_frames: list[CrashFrame] = field(default_factory=list)
    allocated_by_frames: list[CrashFrame] = field(default_factory=list)
    suspicious_frames: list[CrashFrame] = field(default_factory=list)
    rg_evidence: dict[str, list[str]] = field(default_factory=dict)
    confidence: str = "low"
    suggested_checks: list[str] = field(default_factory=list)
    tool_evidence: list[dict] = field(default_factory=list)
    summary: str = ""
    likely_cause: str = ""
    evidence: list[str] = field(default_factory=list)
    exact_matches: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)

    @property
    def suggested_next_steps(self) -> list[str]:
        """Backward-compatible name used by older CLI formatting."""
        return self.suggested_checks

    def to_dict(self) -> dict:
        return {
            "crash_type": self.crash_type,
            "crash_frame": self.crash_frame.to_dict() if self.crash_frame else None,
            "freed_by_frames": [frame.to_dict() for frame in self.freed_by_frames],
            "allocated_by_frames": [frame.to_dict() for frame in self.allocated_by_frames],
            "suspicious_frames": [frame.to_dict() for frame in self.suspicious_frames],
            "rg_evidence": self.rg_evidence,
            "confidence": self.confidence,
            "suggested_checks": self.suggested_checks,
            "tool_evidence": self.tool_evidence,
            "summary": self.summary,
            "likely_cause": self.likely_cause,
            "evidence": self.evidence,
            "relevant_files": self.relevant_files,
        }


def analyze_crash(
    parsed_log,
    results: Iterable[RetrievalResult],
    repo_root: Optional[Path] = None,
    tool_evidence: Optional[Iterable[object]] = None,
) -> CrashTriageReport:
    """Create deterministic crash triage from stack/sanitizer evidence."""
    results = list(results)
    normalized_tool_evidence = _normalize_tool_evidence(tool_evidence)
    raw_log = getattr(parsed_log, "raw_log", "") or ""
    crash_type = _classify_crash(raw_log, getattr(parsed_log, "error_type", "unknown"))
    crash_frames, freed_by_frames, allocated_by_frames = _parse_asan_stacks(
        raw_log,
        fallback_frames=list(getattr(parsed_log, "stack_frames", []) or []),
    )
    crash_frame = crash_frames[0] if crash_frames else None
    suspicious_frames = _score_suspicious_frames(
        crash_type=crash_type,
        crash_frames=crash_frames,
        freed_by_frames=freed_by_frames,
        allocated_by_frames=allocated_by_frames,
        results=results,
    )
    rg_evidence = _rg_evidence_for_frames(repo_root, suspicious_frames[:3])
    confidence = _confidence(crash_type, crash_frame, freed_by_frames, suspicious_frames)
    suggested_checks = _suggested_checks(crash_type, crash_frame, freed_by_frames)
    exact_matches = _flatten_rg_evidence(rg_evidence)
    relevant_files = _top_files(results, crash_frames + freed_by_frames + allocated_by_frames)
    evidence = _build_evidence(
        parsed_log=parsed_log,
        results=results,
        crash_frames=crash_frames,
        freed_by_frames=freed_by_frames,
        allocated_by_frames=allocated_by_frames,
        rg_evidence=rg_evidence,
        tool_evidence=normalized_tool_evidence,
    )
    summary = _summary(crash_type, crash_frame, freed_by_frames, results)
    likely_cause = _likely_cause(crash_type, crash_frame, suspicious_frames)

    return CrashTriageReport(
        crash_type=crash_type,
        crash_frame=crash_frame,
        freed_by_frames=freed_by_frames,
        allocated_by_frames=allocated_by_frames,
        suspicious_frames=suspicious_frames,
        rg_evidence=rg_evidence,
        confidence=confidence,
        suggested_checks=suggested_checks,
        tool_evidence=normalized_tool_evidence,
        summary=summary,
        likely_cause=likely_cause,
        evidence=evidence,
        exact_matches=exact_matches,
        relevant_files=relevant_files,
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


def _parse_asan_stacks(raw_log: str, fallback_frames: list[str]) -> tuple[list[CrashFrame], list[CrashFrame], list[CrashFrame]]:
    """Split ASan/GDB frames into crash, free, and allocation stacks."""
    crash_raw: list[str] = []
    freed_raw: list[str] = []
    allocated_raw: list[str] = []
    current = "crash"

    for line in raw_log.splitlines():
        lower = line.lower()
        if "direct leak of" in lower or "indirect leak of" in lower:
            current = "allocated"
            continue
        if "freed by thread" in lower:
            current = "freed"
            continue
        if "previously allocated by thread" in lower or "allocated by thread" in lower:
            current = "allocated"
            continue
        if not _FRAME_START_RE.search(line.strip()):
            continue
        if current == "freed":
            freed_raw.append(line.strip())
        elif current == "allocated":
            allocated_raw.append(line.strip())
        else:
            crash_raw.append(line.strip())

    if not crash_raw and fallback_frames:
        crash_raw = fallback_frames

    return (
        _parse_frames(crash_raw, "crash"),
        _parse_frames(freed_raw, "freed"),
        _parse_frames(allocated_raw, "allocated"),
    )


def _parse_frames(raw_frames: list[str], evidence_source: str = "crash") -> list[CrashFrame]:
    """Parse symbolized and partial frames without discarding unknown locations."""
    frames: list[CrashFrame] = []
    for raw in raw_frames:
        stripped = raw.strip()
        prefix = _FRAME_PREFIX_RE.search(stripped)
        if not prefix:
            continue
        body = prefix.group("body")
        locations = list(_FRAME_FILE_RE.finditer(body))
        location = locations[-1] if locations else None
        function_text = body[:location.start()] if location else body
        function = _clean_frame_function(function_text)
        frames.append(
            CrashFrame(
                index=int(prefix.group("index")),
                function=function,
                file_path=(location.group("file") if location else "").replace("\\", "/"),
                line_number=int(location.group("line") or 0) if location else 0,
                raw=stripped,
                evidence_source=evidence_source,
            )
        )
    return frames


def _clean_frame_function(value: str) -> str:
    value = _FRAME_ADDRESS_RE.sub("", (value or "").strip())
    if value.startswith("in "):
        value = value[3:]
    value = re.sub(r"\s+at\s*$", "", value).strip()
    value = re.sub(r"\s+\([^)]*=.*$", "", value).strip()
    if value.startswith("operator delete"):
        return "operator delete[]" if "delete[]" in value else "operator delete"
    if value.startswith("operator new"):
        return "operator new[]" if "new[]" in value else "operator new"
    if "(" in value:
        value = value.split("(", 1)[0].strip()
    if not value or value.startswith(("??", "(", "[")):
        return "unknown"
    return value


def _score_suspicious_frames(
    crash_type: str,
    crash_frames: list[CrashFrame],
    freed_by_frames: list[CrashFrame],
    allocated_by_frames: list[CrashFrame],
    results: list[RetrievalResult],
    limit: int = 8,
) -> list[CrashFrame]:
    """Rank frames using explicit evidence rather than file overlap only."""
    result_files = {result.file_path.replace("\\", "/") for result in results}
    candidates: dict[tuple[str, str, int], CrashFrame] = {}

    def add_frame(frame: CrashFrame, source: str) -> None:
        key = (frame.function, frame.file_path, frame.line_number)
        base = 0.1
        if source == "freed":
            base += 0.5
        elif source == "allocated":
            base += 0.1
        elif source == "crash" and frame.index == 0:
            base += 0.0
        elif source == "crash":
            base += 0.1

        if source != "crash" or frame.index != 0:
            base += _keyword_bonus(frame.function)
        if frame.file_path and (
            frame.file_path in result_files or any(path.endswith(frame.file_path) for path in result_files)
        ):
            base += 0.1
        if crash_type == "heap-use-after-free" and source == "freed":
            base += 0.2
        if _is_runtime_frame(frame):
            base -= 0.35

        scored = candidates.get(key)
        score = round(max(0.0, base), 3)
        if scored is None or score > scored.score:
            candidates[key] = CrashFrame(
                index=frame.index,
                function=frame.function,
                file_path=frame.file_path,
                line_number=frame.line_number,
                raw=frame.raw,
                score=score,
                evidence_source=source,
            )

    for frame in crash_frames:
        add_frame(frame, "crash")
    for frame in freed_by_frames:
        add_frame(frame, "freed")
    for frame in allocated_by_frames:
        add_frame(frame, "allocated")

    ranked = sorted(
        candidates.values(),
        key=lambda frame: (-frame.score, frame.index, frame.function),
    )
    return ranked[:limit]


def _keyword_bonus(function: str) -> float:
    tokens = _function_tokens(function)
    score = 0.0
    if any(keyword in tokens for keyword in OWNERSHIP_KEYWORDS):
        score += 0.3
    if any(keyword in tokens for keyword in BOUNDS_KEYWORDS):
        score += 0.2
    return score


def _function_tokens(function: str) -> set[str]:
    """Tokenize qualified/CamelCase names without substring false positives."""
    tokens: set[str] = set()
    for segment in re.split(r"[^A-Za-z0-9]+", function or ""):
        if not segment:
            continue
        tokens.add(segment.lower())
        parts = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            " ",
            segment,
        ).split()
        tokens.update(part.lower() for part in parts)
    return tokens


def _rg_evidence_for_frames(repo_root: Optional[Path], frames: list[CrashFrame]) -> dict[str, list[str]]:
    if repo_root is None:
        return {}
    repo_root = Path(repo_root).resolve()
    evidence: dict[str, list[str]] = {}
    for frame in frames:
        if not frame.function or frame.function in evidence:
            continue
        matches = _search_symbol_files(repo_root, frame.function, max_matches=8)
        if matches:
            evidence[frame.function] = matches
    return evidence


def _search_symbol_files(repo_root: Path, symbol: str, max_matches: int = 8) -> list[str]:
    """Run rg --type cpp -l with a Python fallback for exact symbol evidence."""
    if not symbol:
        return []
    rg = shutil.which("rg")
    if rg:
        try:
            completed = subprocess.run(
                [rg, "--type", "cpp", "--fixed-strings", "-l", symbol, str(repo_root)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            matches = _normalize_rg_file_lines(completed.stdout, repo_root, max_matches=max_matches)
            if matches:
                return matches
        except (OSError, subprocess.SubprocessError):
            pass
    return _fallback_file_search(repo_root, symbol, max_matches=max_matches)


def _normalize_rg_file_lines(stdout: str, repo_root: Path, max_matches: int) -> list[str]:
    files: list[str] = []
    for raw in stdout.splitlines():
        path = Path(raw.strip())
        if not str(path):
            continue
        try:
            display_path = path.resolve().relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            display_path = path.as_posix()
        if display_path not in files:
            files.append(display_path)
        if len(files) >= max_matches:
            break
    return files


def _fallback_file_search(repo_root: Path, symbol: str, max_matches: int) -> list[str]:
    matches: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CPP_SUFFIXES:
            continue
        try:
            if symbol in path.read_text(encoding="utf-8", errors="replace"):
                matches.append(path.relative_to(repo_root).as_posix())
                if len(matches) >= max_matches:
                    return matches
        except OSError:
            continue
    return matches


def _flatten_rg_evidence(rg_evidence: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    for function, files in rg_evidence.items():
        for file_path in files:
            flattened.append(f"{function}: {file_path}")
    return flattened


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


def _build_evidence(
    parsed_log,
    results,
    crash_frames,
    freed_by_frames,
    allocated_by_frames,
    rg_evidence,
    tool_evidence,
) -> list[str]:
    evidence: list[str] = []
    identifiers = list(getattr(parsed_log, "identifiers", []) or [])
    if identifiers:
        evidence.append("Runtime symbols extracted: " + ", ".join(identifiers[:5]))
    if crash_frames:
        evidence.append(f"Crash stack frames parsed: {len(crash_frames)}")
        evidence.append(f"Crash frame: #{crash_frames[0].index} {crash_frames[0].function}")
    if freed_by_frames:
        evidence.append(f"ASan free stack frames parsed: {len(freed_by_frames)}")
    if allocated_by_frames:
        evidence.append(f"ASan allocation stack frames parsed: {len(allocated_by_frames)}")
    if results:
        top = results[0]
        evidence.append(
            f"Top retrieval result: {top.file_path}:{top.start_line} ({top.function_name}) score={top.score:.3f}"
        )
    if rg_evidence:
        evidence.append(f"rg/fallback exact search found evidence for {len(rg_evidence)} suspicious function(s).")
    for item in tool_evidence:
        tool = item.get("tool", "tool")
        status = item.get("status", "unknown")
        evidence.append(f"{tool} evidence collection status: {status}.")
    return evidence


def _normalize_tool_evidence(items: Optional[Iterable[object]]) -> list[dict]:
    normalized: list[dict] = []
    for item in items or []:
        if hasattr(item, "to_dict"):
            value = item.to_dict()
        elif isinstance(item, dict):
            value = dict(item)
        else:
            continue
        normalized.append(value)
    return normalized


def _summary(crash_type: str, crash_frame: Optional[CrashFrame], freed_by_frames: list[CrashFrame], results: list[RetrievalResult]) -> str:
    free_frame = _select_free_frame(freed_by_frames)
    if crash_type == "heap-use-after-free" and free_frame:
        return f"{crash_type} with ASan free-stack evidence near {free_frame.function}."
    if crash_frame:
        return f"{crash_type} triaged around frame #{crash_frame.index} {crash_frame.function}."
    if results:
        top = results[0]
        return f"{crash_type} triaged from retrieval evidence near {top.file_path}:{top.start_line}."
    return f"{crash_type} detected, but no precise code frame was recovered."


def _likely_cause(crash_type: str, crash_frame: Optional[CrashFrame], suspicious_frames: list[CrashFrame]) -> str:
    anchor = ""
    if suspicious_frames:
        anchor = f" Inspect #{suspicious_frames[0].index} {suspicious_frames[0].function} first."
    elif crash_frame:
        anchor = f" Start at crash frame #{crash_frame.index} {crash_frame.function}."

    if crash_type == "heap-use-after-free":
        return "Memory is likely accessed after its lifetime ended." + anchor
    if crash_type in {"heap-buffer-overflow", "stack-buffer-overflow"}:
        return "Code likely reads/writes outside a valid buffer range." + anchor
    if crash_type == "memory_leak":
        return "Allocated memory is likely not released on at least one path." + anchor
    if crash_type == "ubsan_error":
        return "Undefined behavior was reported; inspect the operation and caller-provided values." + anchor
    if crash_type == "segfault":
        return "A null, dangling, or invalid pointer/reference is likely being dereferenced." + anchor
    return "Runtime failure needs inspection of the top crash/retrieval evidence." + anchor


def _confidence(
    crash_type: str,
    crash_frame: Optional[CrashFrame],
    freed_by_frames: list[CrashFrame],
    suspicious_frames: list[CrashFrame],
) -> str:
    actionable_free_frame = _select_user_ownership_frame(freed_by_frames)
    if crash_type == "heap-use-after-free" and actionable_free_frame:
        if _keyword_bonus(actionable_free_frame.function) >= 0.3:
            return "high"
    if crash_frame and not freed_by_frames and len(suspicious_frames) <= 1:
        return "low"
    if crash_type != "unknown" and suspicious_frames and not freed_by_frames:
        return "medium"
    if suspicious_frames:
        return "medium"
    return "low"


def _suggested_checks(
    crash_type: str,
    crash_frame: Optional[CrashFrame],
    freed_by_frames: list[CrashFrame],
) -> list[str]:
    checks: list[str] = []
    if crash_type == "heap-use-after-free":
        free_frame = _select_free_frame(freed_by_frames)
        free_function = free_frame.function if free_frame else "the free/reset path"
        object_type = _infer_object_type(crash_frame, freed_by_frames)
        checks.append(f"Inspect object lifetime of {object_type} after {free_function}")
        if freed_by_frames:
            checks.append(f"Check {free_function} - this is where the object was freed")
        checks.append("Compare the free stack with the later crash stack to find the stale owner or pointer")
    elif crash_type in {"heap-buffer-overflow", "stack-buffer-overflow"}:
        if crash_frame and crash_frame.file_path:
            checks.append(f"Verify index bounds at {crash_frame.file_path}:{crash_frame.line_number}")
        else:
            checks.append("Verify index bounds at the crash frame")
        checks.append("Check caller-provided size, capacity, count, and offset values")
    elif crash_type == "ubsan_error":
        if crash_frame and crash_frame.file_path:
            checks.append(f"Inspect the reported undefined operation at {crash_frame.file_path}:{crash_frame.line_number}")
        else:
            checks.append("Inspect the operation named in the UBSan runtime error")
        checks.append("Validate operand ranges and conversions at the first user-code frame")
    elif crash_type == "memory_leak":
        checks.append("Inspect the first user-code allocation frame and every ownership exit path")
        checks.append("Check early returns, exceptions, and container ownership for a missing release")
    elif crash_type == "segfault":
        checks.append("Compile with -fsanitize=address for more precise information")
        if crash_frame and crash_frame.file_path:
            checks.append(f"Inspect null/dangling pointer assumptions at {crash_frame.file_path}:{crash_frame.line_number}")
    else:
        checks.append("Inspect the crash frame, then move upward through callers until the bad state is introduced")
        checks.append("Rerun with ASan/UBSan/GDB backtrace if this log lacks stack detail")
    return checks[:4]


def _select_free_frame(frames: list[CrashFrame]) -> Optional[CrashFrame]:
    user_frames = [frame for frame in frames if not _is_runtime_frame(frame)]
    ownership_frames = [
        frame for frame in user_frames
        if _keyword_bonus(frame.function) >= 0.3
    ]
    if ownership_frames:
        return ownership_frames[0]
    if user_frames:
        return user_frames[0]
    return None


def _select_user_ownership_frame(frames: list[CrashFrame]) -> Optional[CrashFrame]:
    """Return a non-runtime frame with an ownership signal, if available."""
    return next(
        (
            frame
            for frame in frames
            if not _is_runtime_frame(frame) and _keyword_bonus(frame.function) >= 0.3
        ),
        None,
    )


def _is_runtime_frame(frame: CrashFrame) -> bool:
    function = (frame.function or "").lower()
    path = (frame.file_path or "").replace("\\", "/").lower()
    if function.startswith(("operator delete", "operator new", "__asan", "__ubsan", "__interceptor")):
        return True
    return any(
        marker in path
        for marker in ("libsanitizer", "compiler-rt", "asan_", "ubsan_", "sanitizer_")
    )


def _infer_object_type(crash_frame: Optional[CrashFrame], freed_by_frames: list[CrashFrame]) -> str:
    candidates = []
    if crash_frame:
        candidates.append(crash_frame.function)
    candidates.extend(frame.function for frame in freed_by_frames)
    for function in candidates:
        parts = [part for part in function.split("::") if part]
        if len(parts) >= 2:
            return parts[-2].split("<", 1)[0]
    return "the freed object"
