"""
Code Parser — Tree-sitter based C++ function extractor.

Parses a C++ repository into function-level chunks.
Each chunk represents one function/method with its metadata.

See docs/2_system_architecture.md §1 for spec.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging
import re

import tree_sitter
import tree_sitter_cpp

logger = logging.getLogger(__name__)

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"}

# Patterns that identify test / benchmark files.  Any file whose
# forward-slashed relative path contains one of these substrings is
# excluded from indexing by default.
EXCLUDE_PATTERNS = [
    "_test.cc", "_test.cpp", "_test.h",
    "test_", "tests/", "/test/",
    "_unittest.cc", "_benchmark.cc",
    "_bench.cc", "benchmark/",
]

# Initialise the tree-sitter C++ language and parser once at module level.
_CPP_LANGUAGE = tree_sitter.Language(tree_sitter_cpp.language())
_PARSER = tree_sitter.Parser(_CPP_LANGUAGE)


@dataclass
class Chunk:
    """A single function-level code chunk extracted from a C++ file."""
    chunk_id: str       # "{file_path}::{function_name}"
    file_path: str      # relative to repo root
    function_name: str
    start_line: int
    end_line: int
    code_text: str
    symbol_name: str = ""
    class_name: str = ""
    namespace: str = ""
    signature: str = ""
    language: str = "cpp"


def _is_test_file(rel_path: str) -> bool:
    """Return *True* if *rel_path* matches any EXCLUDE_PATTERNS entry."""
    lower = rel_path.lower()
    return any(pat in lower for pat in EXCLUDE_PATTERNS)


def parse_repository(
    repo_path: Path,
    parser: tree_sitter.Parser = None,
    include_tests: bool = False,
) -> List[Chunk]:
    """Parse all C++ files in a repository into function-level chunks.

    Walks the directory tree, parsing every file whose extension is in
    CPP_EXTENSIONS.  Files that fail to parse are logged and skipped.

    By default, test and benchmark files are excluded (see
    ``EXCLUDE_PATTERNS``).  Set *include_tests* to ``True`` to keep them.

    Args:
        repo_path: Path to the root of the C++ repository.
        parser: Optional pre-initialised tree-sitter Parser.
                Uses the module-level ``_PARSER`` when *None*.
        include_tests: If True, include test/benchmark files.

    Returns:
        List of Chunk objects, one per function found.
    """
    if parser is None:
        parser = _PARSER
    repo_path = Path(repo_path).resolve()
    all_chunks: List[Chunk] = []
    files_parsed = 0
    files_skipped = 0
    files_fallback = 0

    for ext in CPP_EXTENSIONS:
        for file_path in repo_path.rglob(f"*{ext}"):
            rel = str(file_path.relative_to(repo_path)).replace("\\", "/")
            if not include_tests and _is_test_file(rel):
                files_skipped += 1
                continue
            chunks = parse_file(file_path, repo_path, parser=parser)
            if len(chunks) == 1 and chunks[0].function_name == "__file__":
                files_fallback += 1
            all_chunks.extend(chunks)
            files_parsed += 1

    if files_skipped:
        logger.info("Skipped %d test/benchmark files", files_skipped)
    if files_fallback:
        logger.info(
            "%d files fell back to file-level chunks (no functions found)",
            files_fallback,
        )
    logger.info(
        "Parsed %d C++ files, %d chunks (%d function-level, %d file-level)",
        files_parsed,
        len(all_chunks),
        len(all_chunks) - files_fallback,
        files_fallback,
    )
    return all_chunks


def parse_file(
    file_path: Path,
    repo_root: Path,
    parser: tree_sitter.Parser = None,
) -> List[Chunk]:
    """Parse a single C++ file into function-level chunks.

    Uses tree-sitter to build an AST and extracts every
    ``function_definition`` node (including those wrapped in
    ``template_declaration`` nodes).

    Args:
        file_path: Absolute path to the C++ file.
        repo_root: Root of the repository (for computing relative paths).
        parser: Optional pre-initialised tree-sitter Parser.
                Uses the module-level ``_PARSER`` when *None*.

    Returns:
        List of Chunk objects. Empty list if file cannot be parsed or
        is empty.
    """
    if parser is None:
        parser = _PARSER
    file_path = Path(file_path)
    repo_root = Path(repo_root)

    # --- read source bytes -------------------------------------------------
    try:
        source_bytes = file_path.read_bytes()
    except (OSError, IOError) as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    if not source_bytes.strip():
        return []

    # --- parse with tree-sitter --------------------------------------------
    try:
        tree = parser.parse(source_bytes)
    except Exception as exc:
        logger.warning("tree-sitter failed on %s: %s", file_path, exc)
        return []

    # --- compute relative path (forward slashes on Windows) ----------------
    rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

    # --- walk AST and collect function_definition nodes --------------------
    chunks: List[Chunk] = []
    _walk_for_functions(tree.root_node, rel_path, source_bytes, chunks)

    # --- fallback: file-level chunk when no functions found ----------------
    if len(chunks) == 0:
        source_text = source_bytes.decode("utf-8", errors="replace")
        if len(source_text.strip()) > 0:
            logger.debug(
                "No functions found in %s — creating file-level chunk",
                rel_path,
            )
            chunks.append(Chunk(
                chunk_id=f"{rel_path}::__file__",
                file_path=rel_path,
                function_name="__file__",
                start_line=1,
                end_line=len(source_text.splitlines()),
                code_text=source_text[:2000],
                symbol_name="__file__",
                signature=source_text.splitlines()[0].strip() if source_text.splitlines() else "",
                language="cpp",
            ))

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _walk_for_functions(
    node: tree_sitter.Node,
    rel_path: str,
    source_bytes: bytes,
    chunks: List[Chunk],
) -> None:
    """Recursively walk *node* and extract every ``function_definition``.

    Also handles ``template_declaration`` nodes that wrap a
    ``function_definition`` child — the template parameters are ignored
    but the function inside is extracted as a chunk.
    """
    if node.type == "function_definition":
        chunk = _node_to_chunk(node, rel_path, source_bytes)
        if chunk is not None:
            chunks.append(chunk)
        return  # no need to recurse deeper

    if node.type == "template_declaration":
        # Template declarations may contain a function_definition child.
        for child in node.children:
            if child.type == "function_definition":
                chunk = _node_to_chunk(child, rel_path, source_bytes)
                if chunk is not None:
                    chunks.append(chunk)
                return  # found — done with this subtree
        # If no function_definition found (e.g. template class), fall
        # through to recurse normally.

    for child in node.children:
        _walk_for_functions(child, rel_path, source_bytes, chunks)


def _node_to_chunk(
    func_node: tree_sitter.Node,
    rel_path: str,
    source_bytes: bytes,
) -> Optional[Chunk]:
    """Convert a ``function_definition`` AST node into a Chunk."""
    func_name = _extract_function_name(func_node)
    if not func_name:
        return None

    # tree-sitter rows are 0-indexed; we want 1-indexed lines.
    start_line = func_node.start_point.row + 1
    end_line = func_node.end_point.row + 1
    code_text = source_bytes[func_node.start_byte:func_node.end_byte].decode(
        "utf-8", errors="replace"
    )

    namespace, class_name, symbol_name = _build_symbol_metadata(func_name)

    return Chunk(
        chunk_id=f"{rel_path}::{func_name}::L{start_line}",
        file_path=rel_path,
        function_name=func_name,
        start_line=start_line,
        end_line=end_line,
        code_text=code_text,
        symbol_name=symbol_name,
        class_name=class_name,
        namespace=namespace,
        signature=_extract_signature(code_text),
    )


def _extract_function_name(func_node: tree_sitter.Node) -> str:
    """Extract the (possibly class-qualified) function name.

    Handles three cases:
    1. **Out-of-class definition**: ``void Parser::resolve(...)``
       → ``function_declarator > qualified_identifier``
       → returns ``"Parser::resolve"``
    2. **Inline class method**: method defined inside a ``class_specifier``
       → ``function_declarator > field_identifier``
       → prepends enclosing class name → ``"ClassName::method"``
    3. **Plain function**: ``void foo()``
       → ``function_declarator > identifier``
       → returns ``"foo"``
    """
    declarator = _find_child_by_type(func_node, "function_declarator")
    if declarator is None:
        return ""

    # Case 1: qualified_identifier (Parser::resolve)
    qualified = _find_child_by_type(declarator, "qualified_identifier")
    if qualified is not None:
        return qualified.text.decode("utf-8", errors="replace")

    # Case 2: field_identifier (inline class method)
    field_id = _find_child_by_type(declarator, "field_identifier")
    if field_id is not None:
        name = field_id.text.decode("utf-8", errors="replace")
        class_name = _find_enclosing_class(func_node)
        if class_name:
            return f"{class_name}::{name}"
        return name

    # Case 3: plain identifier
    ident = _find_child_by_type(declarator, "identifier")
    if ident is not None:
        return ident.text.decode("utf-8", errors="replace")

    return ""


def _find_enclosing_class(node: tree_sitter.Node) -> Optional[str]:
    """Walk up the parent chain to find an enclosing ``class_specifier``.

    Returns:
        The class name as a string, or None if no enclosing class.
    """
    current = node.parent
    while current is not None:
        if current.type == "class_specifier":
            type_id = _find_child_by_type(current, "type_identifier")
            if type_id is not None:
                return type_id.text.decode("utf-8", errors="replace")
        current = current.parent
    return None


def _find_child_by_type(
    node: tree_sitter.Node, child_type: str
) -> Optional[tree_sitter.Node]:
    """Return the first direct child of *node* with the given type."""
    for child in node.children:
        if child.type == child_type:
            return child
    return None


def _build_symbol_metadata(function_name: str) -> tuple[str, str, str]:
    """Derive namespace/class/symbol metadata from a qualified function name."""
    if not function_name:
        return "", "", ""

    parts = [part for part in function_name.split("::") if part]
    symbol_name = parts[-1] if parts else function_name

    if len(parts) >= 3:
        return "::".join(parts[:-2]), parts[-2], symbol_name
    if len(parts) == 2:
        return "", parts[0], symbol_name
    return "", "", symbol_name


def _extract_signature(code_text: str) -> str:
    """Extract a compact signature-like first line from a function body."""
    lines = [line.strip() for line in code_text.splitlines() if line.strip()]
    if not lines:
        return ""

    signature_parts: List[str] = []
    for line in lines:
        signature_parts.append(line)
        if "{" in line or ")" in line:
            break

    signature = " ".join(signature_parts)
    signature = signature.split("{", 1)[0].strip()
    signature = re.sub(r"\s+", " ", signature)
    return signature
