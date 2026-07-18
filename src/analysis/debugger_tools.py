"""Safe subprocess collectors for native debugger evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Optional


DEFAULT_MAX_OUTPUT_CHARS = 200_000


@dataclass
class ToolEvidence:
    """Observed output and execution metadata from one native tool."""

    tool: str
    status: str
    available: bool
    command: list[str]
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def log_text(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part).strip()


def build_gdb_core_command(gdb: str, binary: Path, core: Path) -> list[str]:
    """Build a deterministic, non-interactive GDB core-dump command."""
    return [
        gdb,
        "--batch",
        "--quiet",
        "--nx",
        "-ex",
        "set pagination off",
        "-ex",
        "set print frame-arguments scalars",
        "-ex",
        "show version",
        "-ex",
        "thread apply all bt",
        "-ex",
        "info registers",
        "-ex",
        "info sharedlibrary",
        str(binary),
        str(core),
    ]


def collect_gdb_core_evidence(
    binary: Path,
    core: Path,
    *,
    gdb_path: Optional[str] = None,
    timeout_seconds: float = 20.0,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ToolEvidence:
    """Collect bounded GDB evidence without executing shell syntax."""
    binary = Path(binary).resolve()
    core = Path(core).resolve()
    gdb = gdb_path or shutil.which("gdb")

    if not gdb:
        return ToolEvidence(
            tool="gdb",
            status="unavailable",
            available=False,
            command=[],
            stderr="GDB was not found on PATH.",
        )

    command = build_gdb_core_command(str(gdb), binary, core)
    missing = [str(path) for path in (binary, core) if not path.is_file()]
    if missing:
        return ToolEvidence(
            tool="gdb",
            status="invalid_input",
            available=True,
            command=command,
            stderr="Missing input file(s): " + ", ".join(missing),
        )

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_cut = _bounded_text(_coerce_text(exc.stdout), max_output_chars)
        stderr, stderr_cut = _bounded_text(_coerce_text(exc.stderr), max_output_chars)
        return ToolEvidence(
            tool="gdb",
            status="timeout",
            available=True,
            command=command,
            stdout=stdout,
            stderr=stderr or f"GDB exceeded the {timeout_seconds:g}s timeout.",
            duration_ms=_elapsed_ms(started),
            truncated=stdout_cut or stderr_cut,
        )
    except OSError as exc:
        return ToolEvidence(
            tool="gdb",
            status="failed",
            available=True,
            command=command,
            stderr=str(exc),
            duration_ms=_elapsed_ms(started),
        )

    stdout, stdout_cut = _bounded_text(completed.stdout, max_output_chars)
    stderr, stderr_cut = _bounded_text(completed.stderr, max_output_chars)
    return ToolEvidence(
        tool="gdb",
        status="ok" if completed.returncode == 0 else "failed",
        available=True,
        command=command,
        return_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=_elapsed_ms(started),
        truncated=stdout_cut or stderr_cut,
    )


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    marker = "\n... DebugAid truncated tool output ...\n"
    remaining = max(0, limit - len(marker))
    head = remaining * 3 // 4
    tail = remaining - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
