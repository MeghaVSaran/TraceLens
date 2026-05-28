"""Utilities for reading and querying C/C++ compilation databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json
import shlex


DEFAULT_COMPILE_COMMANDS_LOCATIONS = (
    "compile_commands.json",
    "build/compile_commands.json",
    "out/compile_commands.json",
)


@dataclass(frozen=True)
class CompileCommandEntry:
    """One entry from compile_commands.json."""

    file_path: str
    directory: str
    command: str
    arguments: tuple[str, ...]
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]
    std_flag: str = ""
    output: str = ""


class CompileCommandsIndex:
    """Small helper index over a compilation database."""

    def __init__(self, entries: Iterable[CompileCommandEntry], repo_root: Optional[Path] = None):
        self.entries = list(entries)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None

    @classmethod
    def from_file(cls, path: Path, repo_root: Optional[Path] = None) -> "CompileCommandsIndex":
        """Load a compilation database from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            raw_entries = json.load(handle)
        entries = [_parse_compile_command_entry(row, repo_root=repo_root) for row in raw_entries]
        return cls(entries, repo_root=repo_root)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def find_best_entry(self, file_hints: Iterable[str]) -> Optional[CompileCommandEntry]:
        """Return the most relevant compile command for a list of source hints."""
        normalized_hints = []
        for hint in file_hints:
            if not isinstance(hint, str):
                continue
            clean = hint.replace("\\", "/").strip()
            if clean:
                normalized_hints.append(clean)
        if not normalized_hints:
            return None

        scored: list[tuple[int, CompileCommandEntry]] = []
        for entry in self.entries:
            entry_path = entry.file_path.replace("\\", "/")
            basename = Path(entry_path).name
            score = 0
            for hint in normalized_hints:
                if hint == entry_path:
                    score = max(score, 100)
                elif entry_path.endswith(f"/{hint}") or entry_path.endswith(hint):
                    score = max(score, 80 + hint.count("/"))
                elif Path(hint).name == basename:
                    score = max(score, 50)
            if score > 0:
                scored.append((score, entry))

        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], len(item[1].file_path)), reverse=True)
        return scored[0][1]

    def covers_file(self, file_hint: str) -> bool:
        """Return True if any compile command matches the given file hint."""
        return self.find_best_entry([file_hint]) is not None


def discover_compile_commands(
    repo_root: Path,
    build_dir: Optional[Path] = None,
    explicit_path: Optional[Path] = None,
) -> Optional[Path]:
    """Find compile_commands.json in common project locations."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if build_dir is not None:
        candidates.append(Path(build_dir) / "compile_commands.json")

    resolved_root = Path(repo_root)
    for relative in DEFAULT_COMPILE_COMMANDS_LOCATIONS:
        candidates.append(resolved_root / relative)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _parse_compile_command_entry(row: dict, repo_root: Optional[Path] = None) -> CompileCommandEntry:
    """Parse one raw compile_commands row into normalized fields."""
    directory = str(row.get("directory", ""))
    command = str(row.get("command", "")).strip()
    arguments = row.get("arguments") or shlex.split(command)
    file_value = str(row.get("file", ""))
    output = str(row.get("output", ""))

    directory_path = Path(directory) if directory else Path(".")
    source_path = Path(file_value)
    if not source_path.is_absolute():
        source_path = (directory_path / source_path).resolve()

    normalized_file_path = _normalize_repo_relative(source_path, repo_root=repo_root)

    include_dirs: list[str] = []
    defines: list[str] = []
    std_flag = ""

    idx = 0
    while idx < len(arguments):
        arg = str(arguments[idx])
        if arg in ("-I", "-isystem") and idx + 1 < len(arguments):
            include_dirs.append(arguments[idx + 1])
            idx += 2
            continue
        if arg.startswith("-I") and len(arg) > 2:
            include_dirs.append(arg[2:])
        elif arg.startswith("-D") and len(arg) > 2:
            defines.append(arg[2:])
        elif arg.startswith("-std="):
            std_flag = arg
        idx += 1

    return CompileCommandEntry(
        file_path=normalized_file_path,
        directory=directory,
        command=command or " ".join(str(a) for a in arguments),
        arguments=tuple(str(a) for a in arguments),
        include_dirs=tuple(include_dirs),
        defines=tuple(defines),
        std_flag=std_flag,
        output=output,
    )


def _normalize_repo_relative(path: Path, repo_root: Optional[Path] = None) -> str:
    """Normalize an absolute path relative to repo root when possible."""
    resolved = path.resolve()
    if repo_root is not None:
        resolved_root = Path(repo_root).resolve()
        try:
            return str(resolved.relative_to(resolved_root)).replace("\\", "/")
        except ValueError:
            pass
    return str(resolved).replace("\\", "/")
