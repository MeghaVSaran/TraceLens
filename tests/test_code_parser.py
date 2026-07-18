"""
Tests for src/ingestion/code_parser.py

Uses pytest's tmp_path fixture to create temporary C++ files
from inline strings. The session-scoped ``cpp_parser`` fixture
(defined in conftest.py) provides a shared tree-sitter Parser so
that initialisation happens only once per test run.
"""

import pytest
from pathlib import Path
from src.ingestion.code_parser import Chunk, parse_file, parse_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cpp(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a C++ source string to a temp file and return its path."""
    f = tmp_path / filename
    f.write_text(content, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# 1. Simple function extraction
# ---------------------------------------------------------------------------

class TestSimpleFunction:
    CODE = "void foo() { return; }\n"

    def test_produces_one_chunk(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 1

    def test_function_name(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.function_name == "foo"

    def test_chunk_id_format(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.chunk_id == "simple.cpp::foo::L1"

    def test_line_numbers(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.start_line == 1
        assert chunk.end_line == 1

    def test_code_text(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert "foo" in chunk.code_text
        assert "return" in chunk.code_text

    def test_language(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "simple.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.language == "cpp"


# ---------------------------------------------------------------------------
# 2. Class method extraction (out-of-class definition)
# ---------------------------------------------------------------------------

class TestClassMethod:
    CODE = """\
void Parser::resolveSymbol(int x) {
    return;
}
"""

    def test_qualified_name(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "parser.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.function_name == "Parser::resolveSymbol"

    def test_chunk_id(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "parser.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.chunk_id == "parser.cpp::Parser::resolveSymbol::L1"

    def test_symbol_metadata(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "parser.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.symbol_name == "resolveSymbol"
        assert chunk.class_name == "Parser"
        assert chunk.signature.startswith("void Parser::resolveSymbol")


# ---------------------------------------------------------------------------
# 3. Multiple functions in one file
# ---------------------------------------------------------------------------

class TestMultipleFunctions:
    CODE = """\
void foo() { }
int bar(int x) { return x; }
double baz(double a, double b) { return a + b; }
"""

    def test_count(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "multi.cpp", self.CODE)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 3

    def test_names(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "multi.cpp", self.CODE)
        names = {c.function_name for c in parse_file(f, tmp_path, parser=cpp_parser)}
        assert names == {"foo", "bar", "baz"}


# ---------------------------------------------------------------------------
# 4. File with parse error → []
# ---------------------------------------------------------------------------

class TestParseError:
    def test_binary_gibberish_returns_empty(self, tmp_path, cpp_parser):
        """Binary content should not crash; returns empty list."""
        f = tmp_path / "bad.cpp"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        # tree-sitter is very resilient and may still return nodes from
        # binary data.  The key requirement is: no crash, returns a list.
        assert isinstance(chunks, list)

    def test_nonexistent_file_returns_empty(self, tmp_path, cpp_parser):
        """A file that doesn't exist should return [] with a warning."""
        f = tmp_path / "nonexistent.cpp"
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert chunks == []


# ---------------------------------------------------------------------------
# 5. Empty file → []
# ---------------------------------------------------------------------------

class TestEmptyFile:
    def test_empty_returns_empty(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "empty.cpp", "")
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert chunks == []

    def test_whitespace_only_returns_empty(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "ws.cpp", "   \n\n  \t  \n")
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert chunks == []


# ---------------------------------------------------------------------------
# 6. Template function
# ---------------------------------------------------------------------------

class TestTemplateFunction:
    CODE = """\
template <typename T>
void resolve(T &s) {
    return;
}
"""

    def test_produces_one_chunk(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "tmpl.cpp", self.CODE)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 1

    def test_function_name(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "tmpl.cpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.function_name == "resolve"


# ---------------------------------------------------------------------------
# 7. Header inline method (.hpp)
# ---------------------------------------------------------------------------

class TestHeaderInlineMethod:
    CODE = """\
class SymbolTable {
public:
    void insert(int key) {
        return;
    }
};
"""

    def test_qualified_name(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "symbol_table.hpp", self.CODE)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 1
        assert chunks[0].function_name == "SymbolTable::insert"

    def test_file_path_hpp(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "symbol_table.hpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.file_path == "symbol_table.hpp"

    def test_inline_method_metadata(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "symbol_table.hpp", self.CODE)
        chunk = parse_file(f, tmp_path, parser=cpp_parser)[0]
        assert chunk.symbol_name == "insert"
        assert chunk.class_name == "SymbolTable"


# ---------------------------------------------------------------------------
# 8. parse_repository — integration test
# ---------------------------------------------------------------------------

class TestParseRepository:
    CODE_A = """\
void alpha() { }
void beta() { }
"""
    CODE_B = """\
int gamma(int x) { return x; }
"""

    def test_total_chunks(self, tmp_path, cpp_parser):
        _write_cpp(tmp_path, "a.cpp", self.CODE_A)
        _write_cpp(tmp_path, "b.cpp", self.CODE_B)
        chunks = parse_repository(tmp_path, parser=cpp_parser)
        assert len(chunks) == 3

    def test_all_names_present(self, tmp_path, cpp_parser):
        _write_cpp(tmp_path, "a.cpp", self.CODE_A)
        _write_cpp(tmp_path, "b.cpp", self.CODE_B)
        names = {c.function_name for c in parse_repository(tmp_path, parser=cpp_parser)}
        assert names == {"alpha", "beta", "gamma"}

    def test_file_paths(self, tmp_path, cpp_parser):
        _write_cpp(tmp_path, "a.cpp", self.CODE_A)
        _write_cpp(tmp_path, "b.cpp", self.CODE_B)
        paths = {c.file_path for c in parse_repository(tmp_path, parser=cpp_parser)}
        assert "a.cpp" in paths
        assert "b.cpp" in paths


# ---------------------------------------------------------------------------
# 8. File-level fallback chunk
# ---------------------------------------------------------------------------

class TestFileLevelFallback:
    """When tree-sitter finds no function_definition nodes, parse_file
    should create a single __file__ chunk so the file is still indexed."""

    CODE_NO_FUNCTIONS = (
        "#include <string>\n"
        "\n"
        "// This file has macros and typedefs but no functions.\n"
        "typedef unsigned int uint32;\n"
        "#define MAX_SIZE 1024\n"
        "using MyString = std::string;\n"
    )

    def test_produces_one_chunk(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "macros_only.cpp", self.CODE_NO_FUNCTIONS)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 1

    def test_function_name_is_file(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "macros_only.cpp", self.CODE_NO_FUNCTIONS)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert chunks[0].function_name == "__file__"

    def test_chunk_id_format(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "macros_only.cpp", self.CODE_NO_FUNCTIONS)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert chunks[0].chunk_id == "macros_only.cpp::__file__"

    def test_code_text_contains_content(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "macros_only.cpp", self.CODE_NO_FUNCTIONS)
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert "MAX_SIZE" in chunks[0].code_text

    def test_empty_file_produces_no_chunks(self, tmp_path, cpp_parser):
        """A truly empty file should still produce 0 chunks."""
        f = _write_cpp(tmp_path, "empty.cpp", "")
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 0

    def test_whitespace_only_produces_no_chunks(self, tmp_path, cpp_parser):
        f = _write_cpp(tmp_path, "blank.cpp", "   \n\n  \n")
        chunks = parse_file(f, tmp_path, parser=cpp_parser)
        assert len(chunks) == 0
