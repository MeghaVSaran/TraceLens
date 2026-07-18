"""Bounded GNU binutils collectors for native symbol evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Iterable, Optional

from src.analysis.debugger_tools import DEFAULT_MAX_OUTPUT_CHARS, ToolEvidence


MAX_SYMBOL_INPUTS = 32
MAX_SYMBOL_RECORDS = 32
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]+$")
_FRAME_ADDRESS_RE = re.compile(r"^\s*#\s*\d+\s+(0x[0-9a-fA-F]+)\b", re.MULTILINE)
_FRAME_SYMBOL_RE = re.compile(
    r"^\s*#\s*\d+\s+(?:0x[0-9a-fA-F]+\s+)?"
    r"(?:in\s+)?([A-Za-z_~][A-Za-z0-9_:<>~*]*)",
    re.MULTILINE,
)
_MANGLED_RE = re.compile(r"(?<![A-Za-z0-9_])(_Z[A-Za-z0-9_.$]+)")
_LOCATION_RE = re.compile(r"^(?P<file>.*?):(?P<line>\d+)(?:\s.*)?$")


@dataclass
class SymbolToolEvidence(ToolEvidence):
    """Native tool execution plus parsed symbol/location records."""

    version: str = ""
    records: list[dict] = field(default_factory=list)


def extract_frame_addresses(text: str, limit: int = MAX_SYMBOL_INPUTS) -> list[str]:
    """Extract unique instruction addresses from GDB-style frame lines."""
    return _unique_limited(_FRAME_ADDRESS_RE.findall(text or ""), limit)


def extract_mangled_symbols(text: str, limit: int = MAX_SYMBOL_INPUTS) -> list[str]:
    """Extract unique Itanium C++ ABI symbols from logs/tool output."""
    return _unique_limited(_MANGLED_RE.findall(text or ""), limit)


def extract_frame_symbols(text: str, limit: int = MAX_SYMBOL_INPUTS) -> list[str]:
    """Extract unique function names from GDB/ASan-style frame lines."""
    return _unique_limited(_FRAME_SYMBOL_RE.findall(text or ""), limit)


def collect_native_symbol_evidence(
    binary: Path,
    debugger_text: str,
) -> list[SymbolToolEvidence]:
    """Run only symbol tools for which the debugger text provides inputs."""
    evidence: list[SymbolToolEvidence] = []
    addresses = extract_frame_addresses(debugger_text)
    frame_symbols = extract_frame_symbols(debugger_text)
    mangled_symbols = extract_mangled_symbols(debugger_text)

    if addresses:
        evidence.append(collect_addr2line_evidence(binary, addresses))
    if frame_symbols:
        evidence.append(collect_nm_evidence(binary, frame_symbols))
    if mangled_symbols:
        evidence.append(collect_cxxfilt_evidence(mangled_symbols))
    return evidence


def symbol_records_text(evidence: Iterable[SymbolToolEvidence]) -> str:
    """Render structured native symbol records as retrieval context."""
    lines: list[str] = []
    for item in evidence:
        for record in item.records:
            display = record.get("display", "").strip()
            if display and display not in lines:
                lines.append(display)
    return "\n".join(lines)


def collect_addr2line_evidence(
    binary: Path,
    addresses: Iterable[str],
    *,
    addr2line_path: Optional[str] = None,
    timeout_seconds: float = 10.0,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SymbolToolEvidence:
    """Map binary addresses to demangled inline-aware source locations."""
    binary = Path(binary).resolve()
    valid_addresses = _unique_limited(
        [address for address in addresses if _ADDRESS_RE.fullmatch(address or "")],
        MAX_SYMBOL_INPUTS,
    )
    tool = addr2line_path or shutil.which("addr2line")
    if not tool:
        return _unavailable("addr2line")
    command = [
        str(tool),
        "-e",
        str(binary),
        "-a",
        "-f",
        "-C",
        "-i",
        *valid_addresses,
    ]
    if not binary.is_file():
        return _invalid_input("addr2line", command, f"Binary does not exist: {binary}")
    if not valid_addresses:
        return _no_input("addr2line", command, "No valid frame addresses were available.")

    evidence = _run_symbol_tool(
        "addr2line",
        command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    evidence.records = _parse_addr2line_records(evidence.stdout)
    return evidence


def collect_nm_evidence(
    binary: Path,
    symbols: Iterable[str],
    *,
    nm_path: Optional[str] = None,
    timeout_seconds: float = 15.0,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SymbolToolEvidence:
    """Confirm requested demangled symbols are defined in a binary."""
    binary = Path(binary).resolve()
    queries = _unique_limited(
        [symbol.strip() for symbol in symbols if symbol and symbol.strip()],
        MAX_SYMBOL_INPUTS,
    )
    tool = nm_path or shutil.which("nm")
    if not tool:
        return _unavailable("nm")
    command = [
        str(tool),
        "--defined-only",
        "--demangle",
        "--line-numbers",
        str(binary),
    ]
    if not binary.is_file():
        return _invalid_input("nm", command, f"Binary does not exist: {binary}")
    if not queries:
        return _no_input("nm", command, "No frame symbols were available.")

    evidence = _run_symbol_tool(
        "nm",
        command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    evidence.records = _parse_nm_records(evidence.stdout, queries)
    return evidence


def collect_cxxfilt_evidence(
    symbols: Iterable[str],
    *,
    cxxfilt_path: Optional[str] = None,
    timeout_seconds: float = 5.0,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SymbolToolEvidence:
    """Demangle bounded Itanium ABI symbols with explicit provenance."""
    queries = _unique_limited(
        [symbol.strip() for symbol in symbols if symbol and symbol.strip()],
        MAX_SYMBOL_INPUTS,
    )
    tool = cxxfilt_path or shutil.which("c++filt")
    if not tool:
        return _unavailable("c++filt")
    command = [str(tool), *queries]
    if not queries:
        return _no_input("c++filt", command, "No mangled C++ symbols were available.")

    evidence = _run_symbol_tool(
        "c++filt",
        command,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    output_lines = evidence.stdout.splitlines()
    evidence.records = [
        {
            "input": symbol,
            "demangled": output_lines[index].strip(),
            "display": f"{symbol} -> {output_lines[index].strip()}",
        }
        for index, symbol in enumerate(queries)
        if index < len(output_lines) and output_lines[index].strip()
    ]
    return evidence


def _run_symbol_tool(
    tool_name: str,
    command: list[str],
    *,
    timeout_seconds: float,
    max_output_chars: int,
) -> SymbolToolEvidence:
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
        return SymbolToolEvidence(
            tool=tool_name,
            status="timeout",
            available=True,
            command=command,
            stdout=stdout,
            stderr=stderr or f"{tool_name} exceeded the {timeout_seconds:g}s timeout.",
            duration_ms=_elapsed_ms(started),
            truncated=stdout_cut or stderr_cut,
            version=_tool_version(command[0]),
        )
    except OSError as exc:
        return SymbolToolEvidence(
            tool=tool_name,
            status="failed",
            available=True,
            command=command,
            stderr=str(exc),
            duration_ms=_elapsed_ms(started),
            version=_tool_version(command[0]),
        )

    stdout, stdout_cut = _bounded_text(completed.stdout, max_output_chars)
    stderr, stderr_cut = _bounded_text(completed.stderr, max_output_chars)
    return SymbolToolEvidence(
        tool=tool_name,
        status="ok" if completed.returncode == 0 else "failed",
        available=True,
        command=command,
        return_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=_elapsed_ms(started),
        truncated=stdout_cut or stderr_cut,
        version=_tool_version(command[0]),
    )


def _parse_addr2line_records(stdout: str) -> list[dict]:
    records: list[dict] = []
    address = ""
    function = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        if _ADDRESS_RE.fullmatch(line):
            address = line
            function = ""
            continue
        if not address:
            continue
        if not function:
            function = line
            continue
        location = _LOCATION_RE.match(line)
        file_path = location.group("file") if location else line
        line_number = int(location.group("line")) if location else 0
        record = {
            "address": address,
            "function": function,
            "file_path": file_path,
            "line_number": line_number,
            "display": f"{address} -> {function} at {file_path}:{line_number}",
        }
        records.append(record)
        function = ""
        if len(records) >= MAX_SYMBOL_RECORDS:
            break
    return records


def _parse_nm_records(stdout: str, queries: list[str]) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for line in stdout.splitlines():
        for query in queries:
            base = query.split("(", 1)[0].strip()
            if not base or not _contains_symbol(line, base):
                continue
            key = (query, line.strip())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "query": query,
                    "definition": line.strip(),
                    "display": f"{query} defined as {line.strip()}",
                }
            )
            if len(records) >= MAX_SYMBOL_RECORDS:
                return records
    return records


def _contains_symbol(line: str, symbol: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
    return re.search(pattern, line) is not None


def _tool_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    output = completed.stdout or completed.stderr
    return output.splitlines()[0].strip() if output else ""


def _unavailable(tool: str) -> SymbolToolEvidence:
    return SymbolToolEvidence(
        tool=tool,
        status="unavailable",
        available=False,
        command=[],
        stderr=f"{tool} was not found on PATH.",
    )


def _invalid_input(tool: str, command: list[str], message: str) -> SymbolToolEvidence:
    return SymbolToolEvidence(
        tool=tool,
        status="invalid_input",
        available=True,
        command=command,
        stderr=message,
        version=_tool_version(command[0]),
    )


def _no_input(tool: str, command: list[str], message: str) -> SymbolToolEvidence:
    return SymbolToolEvidence(
        tool=tool,
        status="no_input",
        available=True,
        command=command,
        stderr=message,
        version=_tool_version(command[0]),
    )


def _unique_limited(values: Iterable[str], limit: int) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
        if len(unique) >= limit:
            break
    return unique


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
