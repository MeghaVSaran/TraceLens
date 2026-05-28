"""Lightweight CMake target/source/dependency indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re
import shlex

FIELD_TOKENS = {
    "NAME", "SRCS", "HDRS", "TEXTUAL_HDRS", "DEPS", "COPTS", "LINKOPTS",
    "DEFINES", "PUBLIC", "PRIVATE", "TESTONLY", "DATA", "TAGS",
}


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
                            for link in target.links:
                                if link not in existing.links:
                                    existing.links.append(link)
                elif lowered in {"absl_cc_library", "absl_cc_test", "absl_cc_binary"}:
                    target = _parse_macro_target(
                        args,
                        command_name=lowered,
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
                            for link in target.links:
                                if link not in existing.links:
                                    existing.links.append(link)
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
        normalized_hint = _normalize_target_token(hint)
        matches = [
            target for target in self.targets
            if target.name.endswith(hint)
            or target.name.endswith(f"_{hint}")
            or _normalize_target_token(target.name) == normalized_hint
            or _normalize_target_token(target.name).endswith(normalized_hint)
        ]
        # Preserve deterministic order and avoid duplicates.
        seen: set[str] = set()
        resolved: list[CMakeTarget] = []
        for target in matches:
            if target.name not in seen:
                seen.add(target.name)
                resolved.append(target)
        return resolved

    def target_reaches_any(self, target: CMakeTarget, providers: list[CMakeTarget]) -> bool:
        """Return True if target transitively links any provider target."""
        provider_names = {_normalize_target_token(provider.name) for provider in providers}
        if not provider_names:
            return False

        seen: set[str] = set()
        queue = [target]
        while queue:
            current = queue.pop(0)
            normalized_current = _normalize_target_token(current.name)
            if normalized_current in provider_names:
                return True
            if normalized_current in seen:
                continue
            seen.add(normalized_current)
            for link in current.links:
                linked_target = self._targets_by_name.get(link)
                if linked_target is not None:
                    queue.append(linked_target)
                    continue
                normalized_link = _normalize_target_token(link)
                for candidate in self.targets:
                    if _normalize_target_token(candidate.name) == normalized_link:
                        queue.append(candidate)
        return False


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


def _parse_macro_target(
    args: str,
    command_name: str,
    cmake_dir: Path,
    repo_root: Path,
    rel_dir: str,
) -> Optional[CMakeTarget]:
    """Parse absl-style macro targets with NAME/SRCS/HDRS/DEPS blocks."""
    tokens = _tokenize_cmake_args(args)
    if not tokens:
        return None

    fields: dict[str, list[str]] = {}
    current_field = ""
    for token in tokens:
        upper = token.upper()
        if upper in FIELD_TOKENS:
            current_field = upper
            fields.setdefault(current_field, [])
            continue
        if current_field:
            fields.setdefault(current_field, []).append(token)

    names = fields.get("NAME", [])
    if not names:
        return None
    name = names[0]
    source_tokens = fields.get("SRCS", []) + fields.get("HDRS", []) + fields.get("TEXTUAL_HDRS", [])
    sources = [
        _resolve_source_token(token, cmake_dir=cmake_dir, repo_root=repo_root)
        for token in source_tokens
        if _looks_like_source(token)
    ]
    sources = [source for source in sources if source]
    links = [_normalize_target_token(token) for token in fields.get("DEPS", []) if _looks_like_target_ref(token)]
    kind = {
        "absl_cc_library": "library",
        "absl_cc_test": "test",
        "absl_cc_binary": "executable",
    }.get(command_name, "macro")
    return CMakeTarget(
        name=name,
        kind=kind,
        defined_in=f"{rel_dir}/CMakeLists.txt" if rel_dir else "CMakeLists.txt",
        sources=sources,
        links=[link for link in links if link],
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
        normalized = _normalize_target_token(token)
        if normalized and normalized not in target.links:
            target.links.append(normalized)


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


def _looks_like_target_ref(token: str) -> bool:
    """Return True for a likely target dependency reference."""
    value = token.strip()
    if not value or value.startswith("$<") or value.startswith("${"):
        return False
    if _looks_like_source(value):
        return False
    return "::" in value or re.match(r"^[A-Za-z0-9_.:+-]+$", value) is not None


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


def _normalize_target_token(value: str) -> str:
    """Normalize target references like absl::foo to local target names."""
    token = str(value or "").strip().strip('"')
    if not token or token.startswith("$<") or token.startswith("${"):
        return ""
    if "::" in token:
        token = token.split("::", 1)[1]
    return token
