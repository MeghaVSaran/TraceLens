"""
Log Parser — Regex-based C++ error log extractor.

Parses raw build/runtime log text into structured ParsedLog objects.
Handles 10 error categories: linker, compiler, include, template, segfault,
ASan, UBSan, memory leak, build-system error, and runtime exception.

See docs/2_system_architecture.md §2 for spec.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)

ERROR_TYPES = {
    "linker_error",
    "compiler_error",
    "include_error",
    "template_error",
    "segfault",
    "asan_error",
    "ubsan_error",
    "memory_leak",
    "build_system_error",
    "runtime_exception",
    "unknown",
}

_TEXT_NORMALIZATION_MAP = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "â€˜": "'",       # mojibake left single quote
    "â€™": "'",       # mojibake right single quote
    "â€œ": '"',       # mojibake left double quote
    "â€�": '"',       # mojibake right double quote
    "\u00a0": " ",   # non-breaking space
}

# ---------------------------------------------------------------------------
# Compiled regex patterns — grouped by error category.
# Order within _ERROR_PATTERNS matters: first match wins for error_type.
# We check most-specific categories first so that e.g. an include error
# is not accidentally classified as a generic compiler error.
# ---------------------------------------------------------------------------

# Identifier character class: captures qualified C++ names like
# llvm::SelectionDAG::getNode  but stops *before* an opening parenthesis
# so that mangled parameter lists are stripped automatically.
_IDENT = r"[A-Za-z_~][A-Za-z0-9_:<>~*]*"

# --- include error ---
_RE_INCLUDE_ERROR = re.compile(
    r"fatal error:\s*(\S+)\s*:\s*No such file or directory",
    re.IGNORECASE,
)

# --- template error ---
_RE_TEMPLATE_ERROR = re.compile(
    r"(?:implicit instantiation of undefined template|undefined template)",
    re.IGNORECASE,
)

# --- linker error ---
_RE_UNDEF_REF = re.compile(
    r"undefined reference to\s+[`'\"]?(?!(?:typeinfo|vtable)\b)(" + _IDENT + r")[`'\"]?",
)
_RE_UNDEF_REF_TYPEINFO = re.compile(
    r"undefined reference to\s+[`'\"]?(?:typeinfo|vtable)\s+for\s+(" + _IDENT + r")[`'\"]?",
)
_RE_MULTI_DEF = re.compile(
    r"multiple definition of\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_INLINE_USED_NOT_DEFINED = re.compile(
    r"inline function\s+[`'\"]?([^`'\"]+)[`'\"]?\s+used but never defined",
    re.IGNORECASE,
)

# --- compiler error ---
_RE_UNDECLARED_IDENT = re.compile(
    r"use of undeclared identifier\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_NO_MATCHING_FN = re.compile(
    r"no matching function for call to\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_NOT_MEMBER_OF = re.compile(
    r"[`'\"]?(" + _IDENT + r")[`'\"]?\s+is not a member of\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_NO_MEMBER_NAMED = re.compile(
    r"has no member named\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_NOT_DECLARED_SCOPE = re.compile(
    r"[`'\"]?(" + _IDENT + r")[`'\"]?\s+was not declared in this scope",
)
_RE_DOES_NOT_NAME_TYPE = re.compile(
    r"[`'\"]?(" + _IDENT + r")[`'\"]?\s+does not name a type",
)
_RE_HAS_NOT_BEEN_DECLARED = re.compile(
    r"[`'\"]?(" + _IDENT + r")[`'\"]?\s+has not been declared",
)
_RE_DECL_OUTSIDE_CLASS = re.compile(
    r"declaration of\s+[`'\"]?(.+?)[`'\"]?\s+outside of class is not definition",
)
_RE_INCOMPLETE_TYPE = re.compile(
    r"invalid use of incomplete type\s+[`'\"]?(.+?)[`'\"]?",
)
_RE_NO_DECLARATION_MATCHES = re.compile(
    r"no declaration matches\s+[`'\"]?([^\n]+)",
)
_RE_FORBIDS_DECLARATION = re.compile(
    r"forbids declaration of\s+[`'\"]?(" + _IDENT + r")[`'\"]?\s+with no type",
    re.IGNORECASE,
)

# --- asan_error ---
_RE_ASAN_HEAP_OVERFLOW = re.compile(
    r"heap-buffer-overflow on address", re.IGNORECASE,
)
_RE_ASAN_USE_AFTER_FREE = re.compile(
    r"use-after-free on address", re.IGNORECASE,
)
_RE_ASAN_STACK_OVERFLOW = re.compile(
    r"stack-buffer-overflow", re.IGNORECASE,
)
_RE_ASAN_GENERIC = re.compile(
    r"AddressSanitizer", re.IGNORECASE,
)
_RE_UBSAN = re.compile(
    r"(?:UndefinedBehaviorSanitizer|runtime error:)", re.IGNORECASE,
)
_RE_LSAN = re.compile(
    r"(?:LeakSanitizer|detected memory leaks)", re.IGNORECASE,
)

# --- build_system_error ---
_RE_CMAKE_FIND = re.compile(
    r"Could not find (?:package|module)\s+([\w:]+)", re.IGNORECASE,
)
_RE_MAKE_NO_RULE = re.compile(
    r"No rule to make target\s+[`'\"]?([^`'\"\s,]+)",
)
_RE_CMAKE_ERROR = re.compile(
    r"CMake Error", re.IGNORECASE,
)

# --- runtime_exception ---
_RE_TERMINATE_THROW = re.compile(
    r"terminate called after throwing an instance of\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_STD_EXCEPTION = re.compile(
    r"(std::(?:bad_alloc|out_of_range|runtime_error|logic_error|invalid_argument|length_error|bad_cast|bad_typeid|overflow_error|underflow_error|domain_error|range_error|bad_weak_ptr|bad_function_call))\b",
)
_RE_WHAT = re.compile(
    r"what\(\):\s*(.+)",
)

# --- segfault / stack trace ---
_RE_SEGFAULT = re.compile(r"Segmentation fault", re.IGNORECASE)
_RE_STACK_FRAME = re.compile(
    r"#\s*(\d+)\s+(?:0x[0-9a-fA-F]+\s+)?(?:in\s+)?(" + _IDENT + r")",
)

# --- file hints ---
_RE_FILE_HINT = re.compile(
    r"([\w.\/\\-]+\.(?:cpp|h|hpp|cc|cxx|c|hxx))\b",
)

# --- source path in error message (absolute path to C++ file with line:col) ---
_RE_SOURCE_PATH = re.compile(
    r"((?:[A-Za-z]:)?[\\/][^\s:]+?\.(?:cc|cpp|h|hpp|c|cxx|hxx)"
    r"|(?:[\w.\-]+[\\/])+[\w.\-]+\.(?:cc|cpp|h|hpp|c|cxx|hxx)):(\d+)(?::(\d+))?:",
)
_RE_OBJECT_PATH = re.compile(
    r"([A-Za-z0-9_./\\-]+\.(?:cc|cpp|h|hpp|c|cxx|hxx))(?:\.pic)?\.o\b"
)
_RE_CMAKE_OBJECT = re.compile(
    r"CMakeFiles[\\/][^\\/]+\.dir[\\/](.+?\.(?:cc|cpp|h|hpp|c|cxx|hxx))(?:\.pic)?\.o\b"
)
_RE_CMAKE_TARGET = re.compile(
    r"CMakeFiles[\\/](?P<target>[^\\/]+)\.dir[\\/](?P<path>.+?\.(?:cc|cpp|h|hpp|c|cxx|hxx))(?:\.pic)?\.o\b"
)
_RE_BUILD_COMPONENT_DIR = re.compile(
    r"(?:^|\b)cd\s+([^\n&;]+?)\s*(?:&&|;|$)"
)

# Ordered list used by extract_error_type — first match wins.
_ERROR_PATTERNS = [
    ("include_error", _RE_INCLUDE_ERROR),
    ("template_error", _RE_TEMPLATE_ERROR),
    ("asan_error", _RE_ASAN_HEAP_OVERFLOW),
    ("asan_error", _RE_ASAN_USE_AFTER_FREE),
    ("asan_error", _RE_ASAN_STACK_OVERFLOW),
    ("memory_leak", _RE_LSAN),
    ("ubsan_error", _RE_UBSAN),
    ("asan_error", _RE_ASAN_GENERIC),
    ("build_system_error", _RE_CMAKE_FIND),
    ("build_system_error", _RE_MAKE_NO_RULE),
    ("build_system_error", _RE_CMAKE_ERROR),
    ("linker_error", _RE_UNDEF_REF),
    ("linker_error", _RE_UNDEF_REF_TYPEINFO),
    ("linker_error", _RE_MULTI_DEF),
    ("linker_error", _RE_DECL_OUTSIDE_CLASS),
    ("linker_error", _RE_INLINE_USED_NOT_DEFINED),
    ("compiler_error", _RE_UNDECLARED_IDENT),
    ("compiler_error", _RE_NO_MATCHING_FN),
    ("compiler_error", _RE_NOT_MEMBER_OF),
    ("compiler_error", _RE_NO_MEMBER_NAMED),
    ("compiler_error", _RE_NOT_DECLARED_SCOPE),
    ("compiler_error", _RE_DOES_NOT_NAME_TYPE),
    ("compiler_error", _RE_HAS_NOT_BEEN_DECLARED),
    ("compiler_error", _RE_INCOMPLETE_TYPE),
    ("compiler_error", _RE_NO_DECLARATION_MATCHES),
    ("compiler_error", _RE_FORBIDS_DECLARATION),
    ("runtime_exception", _RE_TERMINATE_THROW),
    ("runtime_exception", _RE_STD_EXCEPTION),
    ("segfault", _RE_SEGFAULT),
    ("segfault", _RE_STACK_FRAME),
]


@dataclass
class ParsedLog:
    """Structured representation of a parsed C++ error log."""
    raw_log: str
    error_type: str                     # one of ERROR_TYPES
    error_message: str                  # single most informative line
    identifiers: List[str] = field(default_factory=list)   # symbol names
    file_hints: List[str] = field(default_factory=list)    # filenames mentioned
    stack_frames: List[str] = field(default_factory=list)  # segfault frame lines
    source_paths: List[str] = field(default_factory=list)  # normalized file paths from log
    build_targets: List[str] = field(default_factory=list) # target hints from build-system paths


    def query_text(self) -> str:
        """Return text to use as retrieval query (log embedder input).

        Format: error_message + identifiers (with CamelCase expansion)
        + file hints.
        """
        # Expand CamelCase identifiers so "StrCat" becomes
        # "StrCat str cat", giving BM25 more matching surface.
        expanded = list(self.identifiers)
        for ident in self.identifiers:
            parts = re.sub(
                r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])',
                ' ', ident,
            ).split()
            if len(parts) > 1:
                expanded.extend(p.lower() for p in parts if len(p) >= 2)

        parts = [self.error_message] + expanded + self.file_hints + self.build_targets
        return " ".join(parts)


# ---- public API -----------------------------------------------------------

def parse_log(log_text: str, repo_root: Optional[Path] = None) -> ParsedLog:
    """Parse a raw C++ error log into a structured ParsedLog.

    Always returns a ParsedLog even if parsing is partial.

    Args:
        log_text: Raw log string (may be multi-line).

    Returns:
        ParsedLog with all available fields populated.
    """
    normalized_log = _normalize_log_text(log_text)
    error_type = extract_error_type(normalized_log)
    identifiers = extract_identifiers(normalized_log)
    file_hints = extract_file_hints(normalized_log)
    error_message = _pick_error_message(normalized_log, error_type)
    stack_frames = _extract_stack_frames(normalized_log)

    parsed = ParsedLog(
        raw_log=log_text,
        error_type=error_type,
        error_message=error_message,
        identifiers=identifiers,
        file_hints=file_hints,
        stack_frames=stack_frames,
        source_paths=extract_source_paths(normalized_log, repo_root=repo_root),
        build_targets=extract_build_targets(normalized_log),
    )
    logger.debug("Parsed log → type=%s, identifiers=%s", error_type, identifiers)
    return parsed


def extract_error_type(log_text: str) -> str:
    """Classify the primary error type in a log.

    Checks patterns in priority order (most-specific first) and returns
    the category of the first match.

    Args:
        log_text: Raw log string.

    Returns:
        One of: linker_error, compiler_error, include_error,
                template_error, segfault, asan_error, ubsan_error,
                memory_leak, build_system_error, runtime_exception, unknown.
    """
    log_text = _normalize_log_text(log_text)
    for error_type, pattern in _ERROR_PATTERNS:
        if pattern.search(log_text):
            return error_type
    return "unknown"


def extract_identifiers(log_text: str) -> List[str]:
    """Extract C++ symbol names from a log.

    Looks for patterns like:
    - 'undefined reference to `Parser::resolveSymbol`'
    - 'use of undeclared identifier resolveSymbol'
    - 'multiple definition of SymbolTable::insert'
    - 'no matching function for call to Parser::parse'
    - '#0  Parser::resolveSymbol' (stack frames)

    The regex capture groups intentionally stop before '(' so that
    C++ mangled parameter lists are stripped automatically.

    Args:
        log_text: Raw log string.

    Returns:
        List of unique identifier strings found (order-preserved).
    """
    log_text = _normalize_log_text(log_text)
    seen: set = set()
    result: List[str] = []

    # All identifier-bearing patterns with their compiled regexes
    ident_patterns = [
        _RE_UNDEF_REF,
        _RE_UNDEF_REF_TYPEINFO,
        _RE_MULTI_DEF,
        _RE_DECL_OUTSIDE_CLASS,
        _RE_INLINE_USED_NOT_DEFINED,
        _RE_UNDECLARED_IDENT,
        _RE_NO_MATCHING_FN,
        _RE_NOT_MEMBER_OF,
        _RE_NO_MEMBER_NAMED,
        _RE_NOT_DECLARED_SCOPE,
        _RE_DOES_NOT_NAME_TYPE,
        _RE_HAS_NOT_BEEN_DECLARED,
        _RE_NO_DECLARATION_MATCHES,
        _RE_FORBIDS_DECLARATION,
        _RE_STACK_FRAME,
        _RE_TERMINATE_THROW,
        _RE_STD_EXCEPTION,
        _RE_CMAKE_FIND,
        _RE_MAKE_NO_RULE,
    ]

    for pattern in ident_patterns:
        for match in pattern.finditer(log_text):
            # For stack frames the identifier is group(2), else group(1)
            if pattern is _RE_STACK_FRAME:
                ident = match.group(2)
            else:
                ident = match.group(1)

            # Strip surrounding quotes / backticks that may have leaked
            ident = ident.strip("`'\"")

            # Strip synthetic __BOGUS_ prefix to recover real symbol name
            ident = _clean_identifier(ident)

            if ident and ident not in seen:
                seen.add(ident)
                result.append(ident)

    return result


def extract_file_hints(log_text: str) -> List[str]:
    """Extract filenames mentioned in a log.

    Catches both general C/C++ filenames and the specific include-error
    pattern (``fatal error: X.h: No such file``).

    Args:
        log_text: Raw log string.

    Returns:
        List of unique filename strings (e.g. ['resolve.cpp', 'symbol_table.h']).
    """
    log_text = _normalize_log_text(log_text)
    seen: set = set()
    result: List[str] = []

    # Include-error specific extraction first (higher quality)
    for m in _RE_INCLUDE_ERROR.finditer(log_text):
        fname = m.group(1)
        if fname not in seen:
            seen.add(fname)
            result.append(fname)

    # General C/C++ filename extraction
    for m in _RE_FILE_HINT.finditer(log_text):
        fname = m.group(1)
        if fname not in seen:
            seen.add(fname)
            result.append(fname)

    return result


def extract_build_targets(log_text: str) -> List[str]:
    """Extract CMake target hints from object-file paths in logs."""
    log_text = _normalize_log_text(log_text)
    seen: set[str] = set()
    result: List[str] = []
    for match in _RE_CMAKE_TARGET.finditer(log_text):
        target = match.group("target").strip()
        if target and target not in seen:
            seen.add(target)
            result.append(target)
    return result


# ---- internal helpers ------------------------------------------------------

def _pick_error_message(log_text: str, error_type: str) -> str:
    """Select the single most informative line from the log.

    Scans all lines and returns the first one that matches a pattern
    associated with the detected *error_type*.  Falls back to the first
    non-empty line.

    Args:
        log_text: Raw log text.
        error_type: Already-detected error category.

    Returns:
        A single-line string (stripped).
    """
    # Map error_type to its relevant patterns
    type_to_patterns = {
        "include_error": [_RE_INCLUDE_ERROR],
        "template_error": [_RE_TEMPLATE_ERROR],
        "linker_error": [
            _RE_UNDEF_REF,
            _RE_UNDEF_REF_TYPEINFO,
            _RE_MULTI_DEF,
            _RE_DECL_OUTSIDE_CLASS,
            _RE_INLINE_USED_NOT_DEFINED,
        ],
        "compiler_error": [
            _RE_UNDECLARED_IDENT, _RE_NO_MATCHING_FN,
            _RE_NOT_MEMBER_OF, _RE_NO_MEMBER_NAMED,
            _RE_NOT_DECLARED_SCOPE, _RE_DOES_NOT_NAME_TYPE,
            _RE_HAS_NOT_BEEN_DECLARED, _RE_INCOMPLETE_TYPE,
            _RE_NO_DECLARATION_MATCHES, _RE_FORBIDS_DECLARATION,
        ],
        "segfault": [_RE_SEGFAULT, _RE_STACK_FRAME],
        "asan_error": [
            _RE_ASAN_HEAP_OVERFLOW, _RE_ASAN_USE_AFTER_FREE,
            _RE_ASAN_STACK_OVERFLOW, _RE_ASAN_GENERIC,
        ],
        "ubsan_error": [_RE_UBSAN],
        "memory_leak": [_RE_LSAN],
        "build_system_error": [_RE_CMAKE_FIND, _RE_MAKE_NO_RULE, _RE_CMAKE_ERROR],
        "runtime_exception": [_RE_TERMINATE_THROW, _RE_STD_EXCEPTION, _RE_WHAT],
    }

    patterns = type_to_patterns.get(error_type, [])
    first_nonempty = ""

    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not first_nonempty:
            first_nonempty = stripped

        for pat in patterns:
            if pat.search(stripped):
                return stripped

    return first_nonempty


def _extract_stack_frames(log_text: str) -> List[str]:
    """Extract stack frame lines from the log (e.g. from segfault traces).

    Args:
        log_text: Raw log text.

    Returns:
        List of raw stack-frame lines, in order.
    """
    frames: List[str] = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if _RE_STACK_FRAME.search(stripped):
            frames.append(stripped)
    return frames


def _clean_identifier(ident: str) -> str:
    """Clean up an extracted identifier.

    Strips the synthetic ``__BOGUS_`` prefix used by
    ``generate_synthetic_errors.py`` to recover the real symbol name.
    """
    ident = (ident or "").strip().strip("`'\"")
    if not ident:
        return ""

    # Linker declaration/warning lines may include return types + signatures.
    if "(" in ident and ("::" in ident or " " in ident):
        match = re.search(r"([A-Za-z_~][A-Za-z0-9_:<>~*]*)\s*\(", ident)
        if match:
            candidate = match.group(1).strip()
            if "::" in candidate or len(candidate) >= 2:
                ident = candidate

    # Remove typeinfo/vtable prefix noise and qualifiers.
    ident = re.sub(r"^(?:typeinfo|vtable)\s+for\s+", "", ident).strip()
    ident = re.sub(
        r"\b(?:const|volatile|static|inline|virtual|explicit|friend|extern|"
        r"mutable|constexpr|typename|class|struct|enum|unsigned|signed|long|"
        r"short|bool|int|void)\b",
        " ",
        ident,
    )
    ident = re.sub(r"\s+", " ", ident).strip()
    if " " in ident:
        ident = ident.split(" ")[-1]

    # Synthetic corruption marker appears in generated samples.
    ident = ident.replace("__BOGUS_", "")
    return ident.strip("`'\" ")


def extract_source_paths(
    log_text: str,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Extract and normalize source file paths embedded in error messages.

    Many compiler error messages include the absolute path to the file that
    caused the error, e.g.:

        /tmp/abseil/absl/strings/str_cat.cc:143:16: error: ...

    This function extracts those paths and attempts to normalize them to
    relative paths within the indexed repository.

    Normalization strategy (no hardcoded repo names):
      1. For each absolute path found, try every suffix starting from each
         path component (e.g., for ``/tmp/abseil/absl/strings/str_cat.cc``
         try ``absl/strings/str_cat.cc``, ``strings/str_cat.cc``, ...).
      2. If ``repo_root`` is given, check whether that suffix exists as a
         file under ``repo_root``. Accept the *longest* matching suffix.
      3. If no suffix matches (or ``repo_root`` is ``None``), fall back to
         the bare filename (e.g., ``str_cat.cc``).

    Args:
        log_text: Raw log string.
        repo_root: Optional path to the indexed repository root.  When
            provided, enables suffix-matching normalization.

    Returns:
        Sorted list of unique normalized relative paths.
    """
    log_text = _normalize_log_text(log_text)
    raw_paths: set = set()
    for match in _RE_SOURCE_PATH.finditer(log_text):
        raw_paths.add(match.group(1))
    for match in _RE_OBJECT_PATH.finditer(log_text):
        raw_paths.add(match.group(1))
    for match in _RE_CMAKE_OBJECT.finditer(log_text):
        raw_paths.add(match.group(1))
    for match in _RE_FILE_HINT.finditer(log_text):
        candidate = match.group(1)
        if "/" in candidate or "\\" in candidate:
            raw_paths.add(candidate)

    component_prefixes = _extract_build_component_prefixes(log_text, repo_root=repo_root)

    normalized: set = set()
    for raw in raw_paths:
        raw_slash = raw.replace("\\", "/")
        if raw_slash.startswith("./"):
            raw_slash = raw_slash[2:]
        parts = Path(raw_slash).parts
        best: Optional[str] = None
        is_rooted = bool(parts) and (
            parts[0] in ("\\", "/") or bool(re.match(r"^[A-Za-z]:$", parts[0]))
        )

        if repo_root is not None:
            resolved_root = Path(repo_root).resolve()
            direct_candidate = resolved_root / raw_slash
            if raw_slash and not is_rooted and direct_candidate.exists():
                best = raw_slash

            if best is None:
                # Try successively shorter suffixes of the path.
                # For rooted absolute paths skip the root token.
                start_idx = 1 if is_rooted else 0
                for i in range(start_idx, len(parts)):
                    suffix = Path(*parts[i:])
                    candidate = resolved_root / suffix
                    if candidate.exists():
                        # Keep the longest (most specific) match
                        rel = str(suffix).replace("\\", "/")
                        if best is None or len(rel) > len(best):
                            best = rel

            # Build-system context expansion for mirrored build directories:
            # e.g. "internal/flag.cc" + "absl/flags" -> "absl/flags/internal/flag.cc".
            if best is None and component_prefixes and raw_slash:
                for prefix in component_prefixes:
                    prefixed = f"{prefix.rstrip('/')}/{raw_slash.lstrip('/')}"
                    candidate = resolved_root / prefixed
                    if candidate.exists():
                        best = prefixed
                        break

        if best is None:
            normalized_raw = raw_slash
            if not is_rooted and "/" in normalized_raw:
                best = normalized_raw
            else:
                # Fallback: use just the filename for BM25 matching
                best = Path(raw_slash).name

        normalized.add(best)

    return sorted(normalized)


def _extract_build_component_prefixes(
    log_text: str,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Infer source-tree prefixes from build command working directories.

    CMake/Ninja logs often enter a mirrored build directory before showing an
    object path, for example ``cd /tmp/build/project/src/core && ...`` followed
    by ``CMakeFiles/app.dir/internal/foo.cc.o``.  The object path alone may be
    too short, so we try suffixes of the working directory as candidate source
    prefixes and later validate them against ``repo_root``.
    """
    prefixes: List[str] = []
    resolved_root = Path(repo_root).resolve() if repo_root is not None else None

    for match in _RE_BUILD_COMPONENT_DIR.finditer(log_text):
        cwd = match.group(1).strip().strip('"').strip("'")
        cwd = cwd.replace("\\", "/").rstrip("/")
        if not cwd:
            continue

        candidates: List[str] = []
        cwd_path = Path(cwd)
        if resolved_root is not None:
            try:
                rel = cwd_path.resolve().relative_to(resolved_root)
                candidates.append(str(rel).replace("\\", "/"))
            except (OSError, ValueError):
                pass

        parts = [part for part in cwd.split("/") if part and part not in {".", ".."}]
        for index in range(len(parts)):
            candidates.append("/".join(parts[index:]))

        for candidate in candidates:
            candidate = candidate.strip("/")
            if candidate and candidate not in prefixes:
                prefixes.append(candidate)

    return prefixes


def _normalize_log_text(log_text: str) -> str:
    """Normalize punctuation/quotes so regex extraction is robust."""
    normalized = log_text or ""
    for src, dst in _TEXT_NORMALIZATION_MAP.items():
        normalized = normalized.replace(src, dst)
    return normalized
