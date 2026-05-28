"""Lightweight CMake target/source/dependency indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re
import shlex


@dataclass
class CMakeTarget:
    """One parsed CMake target."""

    name: str
    kind: str
    defined_in: str
    sources: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


class CMakeProjectIndex:
    """Index targets and their source/link relationships from CMakeLists."""

    def __init__(self, repo_root: Path, targets: list[CMakeTarget]):
        self.repo_root = Path(repo_root).resolve()
        self.targets = targets
        self._targets_by_name = {target.name: target for target in targets}
        self._source_to_targets: dict[str, list[CMakeTarget]] = {}
        for target in targets:
            for source in target.sources:
                self._source_to_targets.setdefault(source, []).append(target)

    @classmethod
    def from_repo(cls, repo_root: Path) -> "CMakeProjectIndex":
        """Parse all CMakeLists.txt files under the repository root."""
        repo_root = Path(repo_root).resolve()
        targets: dict[str, CMakeTarget] = {}
        for cmake_file in sorted(repo_root.rglob("CMakeLists.txt")):
            rel_dir = cmake_file.parent.relative_to(repo_root)
            rel_dir_str = str(rel_dir).replace("\\", "/")
            text = cmake_file.read_text(encoding="utf-8", errors="replace")
            for command_name, args in _extract_cmake_commands(text):
                lowered = command_name.lower()
                if lowered in {"add_library", "add_executable"}:
                    target = _parse_add_target(
                        args,
                        kind=lowered.replace("add_", ""),
                        cmake_dir=cmake_file.parent,
                        repo_root=repo_root,
                        rel_dir=rel_dir_str,
                    )
                    if target is not None:
                        existing = targets.get(target.name)
                        if existing is None:
                            targets[target.name] = target
                        else:
                            for source in target.sources:
                                if source not in existing.sources:
                                    existing.sources.append(source)
                elif lowered == "target_link_libraries":
                    _apply_target_links(args, targets)
        return cls(repo_root, list(targets.values()))

    def __bool__(self) -> bool:
        return bool(self.targets)

    def get_targets_for_file(self, file_path: str) -> list[CMakeTarget]:
        """Return targets that directly mention a source file."""
        normalized = _normalize_rel(file_path)
        return list(self._source_to_targets.get(normalized, []))

    def get_target(self, name: str) -> Optional[CMakeTarget]:
        """Return a target by exact name."""
        return self._targets_by_name.get(name)

    def find_targets_by_hint(self, hint: str) -> list[CMakeTarget]:
        """Resolve a build-target hint against known target names."""
        hint = str(hint or "").strip()
        if not hint:
            return []
        exact = self.get_target(hint)
        if exact is not None:
            return [exact]
        matches = [
            target for target in self.targets
            if target.name.endswith(hint) or target.name.endswith(f"_{hint}")
        ]
        # Preserve deterministic order and avoid duplicates.
        seen: set[str] = set()
        resolved: list[CMakeTarget] = []
        for target in matches:
            if target.name not in seen:
                seen.add(target.name)
                resolved.append(target)
        return resolved


def _extract_cmake_commands(text: str) -> list[tuple[str, str]]:
    """Extract top-level cmake commands with their raw argument strings."""
    commands: list[tuple[str, str]] = []
    idx = 0
    n = len(text)
    while idx < n:
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text[idx:])
        if match is None:
            break
        command = match.group(1)
        open_paren = idx + match.end() - 1
        depth = 0
        end = open_paren
        while end < n:
            char = text[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if end >= n:
            break
        args = text[open_paren + 1:end]
        commands.append((command, _strip_cmake_comments(args)))
        idx = end + 1
    return commands


def _parse_add_target(
    args: str,
    kind: str,
    cmake_dir: Path,
    repo_root: Path,
    rel_dir: str,
) -> Optional[CMakeTarget]:
    """Parse add_library/add_executable arguments into a target."""
    tokens = _tokenize_cmake_args(args)
    if not tokens:
        return None
    name = tokens[0]
    source_tokens = []
    for token in tokens[1:]:
        upper = token.upper()
        if upper in {"STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "ALIAS", "EXCLUDE_FROM_ALL", "IMPORTED"}:
            continue
        if _looks_like_source(token):
            source_tokens.append(token)
    sources = [
        _resolve_source_token(token, cmake_dir=cmake_dir, repo_root=repo_root)
        for token in source_tokens
    ]
    sources = [source for source in sources if source]
    return CMakeTarget(
        name=name,
        kind=kind,
        defined_in=f"{rel_dir}/CMakeLists.txt" if rel_dir else "CMakeLists.txt",
        sources=sources,
    )


def _apply_target_links(args: str, targets: dict[str, CMakeTarget]) -> None:
    """Parse target_link_libraries and attach dependencies."""
    tokens = _tokenize_cmake_args(args)
    if not tokens:
        return
    target_name = tokens[0]
    target = targets.get(target_name)
    if target is None:
        target = CMakeTarget(name=target_name, kind="unknown", defined_in="unknown")
        targets[target_name] = target
    for token in tokens[1:]:
        upper = token.upper()
        if upper in {"PRIVATE", "PUBLIC", "INTERFACE", "LINK_PRIVATE", "LINK_PUBLIC"}:
            continue
        if token.startswith("$<"):
            continue
        if token not in target.links:
            target.links.append(token)


def _tokenize_cmake_args(args: str) -> list[str]:
    """Split a cmake command argument string into shell-like tokens."""
    cleaned = re.sub(r"[\r\n\t]+", " ", args)
    try:
        return shlex.split(cleaned, posix=True)
    except ValueError:
        return cleaned.split()


def _strip_cmake_comments(text: str) -> str:
    """Remove # comments from cmake argument text."""
    lines = []
    for line in text.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _looks_like_source(token: str) -> bool:
    """Return True for source/header-like tokens."""
    value = token.strip()
    return value.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"))


def _resolve_source_token(token: str, cmake_dir: Path, repo_root: Path) -> str:
    """Resolve a source token relative to the CMakeLists location."""
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = (cmake_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        return _normalize_rel(candidate.relative_to(repo_root))
    except ValueError:
        return ""


def _normalize_rel(value) -> str:
    """Normalize a relative path to forward-slash form."""
    return str(value).replace("\\", "/").lstrip("./")
