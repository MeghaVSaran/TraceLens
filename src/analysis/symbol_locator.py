"""Lightweight symbol declaration/definition lookup for C/C++ repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import re


CPP_GLOB_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx")
CALL_OPERATOR_PREFIXES = ("->", ".")


@dataclass(frozen=True)
class SymbolLocation:
    """Represents one declaration or definition site for a symbol."""

    file_path: str
    line_number: int
    signature: str
    kind: str


def locate_symbol_sites(
    repo_root: Path,
    symbol: str,
    candidate_files: Optional[Iterable[str]] = None,
) -> dict[str, Optional[SymbolLocation]]:
    """Locate declaration and definition sites for a symbol in a repo."""
    repo_root = Path(repo_root).resolve()
    normalized_symbol = (symbol or "").strip()
    if not normalized_symbol:
        return {"declaration": None, "definition": None}

    files_to_scan = _gather_files(repo_root, candidate_files=candidate_files)
    declaration: Optional[SymbolLocation] = None
    definition: Optional[SymbolLocation] = None

    for file_path in files_to_scan:
        match = _scan_file_for_symbol(repo_root, file_path, normalized_symbol)
        if match is None:
            continue
        if match.kind == "declaration" and declaration is None:
            declaration = match
        elif match.kind == "definition" and definition is None:
            definition = match
        if declaration is not None and definition is not None:
            break

    return {
        "declaration": declaration,
        "definition": definition,
    }


def _gather_files(repo_root: Path, candidate_files: Optional[Iterable[str]]) -> list[Path]:
    """Build an ordered file list, preferring candidate files first."""
    all_files = []
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CPP_GLOB_SUFFIXES:
            all_files.append(path.resolve())

    if not candidate_files:
        return sorted(all_files)

    preferred: list[Path] = []
    remaining: list[Path] = []
    normalized_candidates = [str(candidate).replace("\\", "/") for candidate in candidate_files]
    candidate_set = set(normalized_candidates)

    for path in all_files:
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        if rel in candidate_set:
            preferred.append(path)
        else:
            remaining.append(path)

    return preferred + sorted(remaining)


def _scan_file_for_symbol(repo_root: Path, file_path: Path, symbol: str) -> Optional[SymbolLocation]:
    """Return the first declaration/definition found in one file."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    qualifiers = [part for part in symbol.split("::")[:-1] if part]
    tail = symbol.split("::")[-1]
    if not tail:
        return None

    tail_pattern = re.compile(rf"\b{re.escape(tail)}\b")

    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or tail_pattern.search(stripped) is None:
            continue
        if "(" not in stripped or ")" not in stripped:
            continue
        if not _looks_like_symbol_declaration_or_definition(stripped, tail):
            continue

        after_paren = stripped.split(")", 1)[-1]
        kind = None
        if "{" in after_paren:
            kind = "definition"
        elif ";" in after_paren:
            kind = "declaration"
        if kind is None:
            continue

        rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
        return SymbolLocation(
            file_path=rel,
            line_number=line_number,
            signature=stripped,
            kind=kind,
        )
    return None


def _looks_like_symbol_declaration_or_definition(line: str, tail: str) -> bool:
    """Reject call sites and assertions that merely invoke the symbol."""
    match = re.search(rf"\b{re.escape(tail)}\b\s*\(", line)
    if match is None:
        return False

    prefix = line[:match.start()].rstrip()
    if prefix.endswith(CALL_OPERATOR_PREFIXES):
        return False
    if prefix.endswith(("::", "~")):
        return True

    # Declarations/definitions typically have a return type, qualifier, or
    # class scope immediately before the symbol rather than a macro argument.
    if not prefix:
        return False
    tokens = re.split(r"\s+", prefix)
    last_token = tokens[-1] if tokens else ""
    if last_token in {"return", "if", "while", "for", "switch", "case"}:
        return False
    if "(" in prefix and not prefix.endswith(")"):
        return False
    return True
