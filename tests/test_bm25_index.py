"""
Tests for src/indexing/bm25_index.py

Tests build, query, empty-match, and save/load round-trip
using synthetic C++ code strings and a FakeChunk dataclass.
"""

import pytest
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from src.indexing.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeChunk:
    """Minimal Chunk-like object for testing."""
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    code_text: str
    symbol_name: str = ""
    class_name: str = ""
    namespace: str = ""
    signature: str = ""


CHUNKS = [
    FakeChunk(
        chunk_id="parser.cpp::Parser::resolveSymbol",
        file_path="parser.cpp",
        function_name="Parser::resolveSymbol",
        start_line=10,
        code_text=(
            "void Parser::resolveSymbol(Symbol &s) {\n"
            "    SymbolTable *table = getTable();\n"
            "    table->lookup(s.name());\n"
            "}"
        ),
    ),
    FakeChunk(
        chunk_id="lexer.cpp::Lexer::nextToken",
        file_path="lexer.cpp",
        function_name="Lexer::nextToken",
        start_line=50,
        code_text=(
            "Token Lexer::nextToken() {\n"
            "    char c = peek();\n"
            "    if (isdigit(c)) return scanNumber();\n"
            "    return scanIdentifier();\n"
            "}"
        ),
    ),
    FakeChunk(
        chunk_id="main.cpp::main",
        file_path="main.cpp",
        function_name="main",
        start_line=1,
        code_text=(
            "int main(int argc, char **argv) {\n"
            "    Parser parser;\n"
            "    parser.parse(argv[1]);\n"
            "    return 0;\n"
            "}"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildAndQuery:
    """Build index and query with a matching identifier."""

    def test_query_returns_results(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("resolveSymbol", top_k=3)
        assert len(results) > 0

    def test_top_result_matches_identifier(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("resolveSymbol SymbolTable lookup", top_k=3)
        assert results[0]["chunk_id"] == "parser.cpp::Parser::resolveSymbol"

    def test_result_fields(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("nextToken", top_k=1)
        r = results[0]
        assert "chunk_id" in r
        assert "file_path" in r
        assert "function_name" in r
        assert "start_line" in r
        assert "score" in r
        assert isinstance(r["score"], float)

    def test_top_k_respected(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("parser", top_k=2)
        assert len(results) <= 2


class TestNoMatch:
    """Query that produces no matches returns empty list."""

    def test_unrelated_query(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("completelyUnrelatedXyzzyFoo42", top_k=5)
        assert results == []

    def test_empty_query(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query("", top_k=5)
        assert results == []

    def test_query_before_build(self):
        idx = BM25Index()
        results = idx.query("resolveSymbol", top_k=5)
        assert results == []


class TestSaveLoad:
    """Pickle round-trip preserves query results."""

    def test_round_trip(self, tmp_path):
        idx = BM25Index()
        idx.build(CHUNKS)
        original = idx.query("resolveSymbol", top_k=3)

        save_path = tmp_path / "bm25.pkl"
        idx.save(save_path)

        idx2 = BM25Index()
        idx2.load(save_path)
        restored = idx2.query("resolveSymbol", top_k=3)

        assert len(restored) == len(original)
        assert restored[0]["chunk_id"] == original[0]["chunk_id"]
        assert abs(restored[0]["score"] - original[0]["score"]) < 1e-6

    def test_file_exists_after_save(self, tmp_path):
        idx = BM25Index()
        idx.build(CHUNKS)
        save_path = tmp_path / "subdir" / "bm25.pkl"
        idx.save(save_path)
        assert save_path.exists()


class TestTokenize:
    """Verify the tokenizer handles C++ constructs correctly."""

    def test_splits_operators(self):
        idx = BM25Index()
        tokens = idx._tokenize("table->lookup(s.name())")
        # Should contain 'table', 'lookup', 'name' but not '->' or '(' etc.
        assert "table" in tokens
        assert "lookup" in tokens
        assert "name" in tokens
        assert "->" not in tokens

    def test_filters_short_tokens(self):
        idx = BM25Index()
        tokens = idx._tokenize("int x = 0;")
        assert "x" not in tokens  # too short (1 char)

    def test_filters_numeric_tokens(self):
        idx = BM25Index()
        tokens = idx._tokenize("arr[42] = 100;")
        assert "42" not in tokens
        assert "100" not in tokens

    def test_lowercases(self):
        idx = BM25Index()
        tokens = idx._tokenize("Parser::ResolveSymbol")
        assert all(t == t.lower() for t in tokens)


class TestSymbolQuery:
    """Symbol lookup should provide exact-candidate retrieval."""

    def test_query_by_symbols_returns_exact_match(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query_by_symbols(["Parser::resolveSymbol"], top_k=3)
        assert results
        assert results[0]["chunk_id"] == "parser.cpp::Parser::resolveSymbol"

    def test_negative_scores_not_dropped(self):
        """Regression: BM25 can produce negative scores for common terms."""
        idx = BM25Index()
        idx._chunks = CHUNKS

        class _FakeBM25:
            @staticmethod
            def get_scores(_):
                return np.array([-0.2, -0.5, -0.8], dtype=float)

        idx._index = _FakeBM25()
        results = idx.query("parser", top_k=2)
        assert len(results) == 2


class TestPathPrefixQuery:
    """Path-prefix lookup should recover component-level candidates."""

    def test_query_by_path_prefix(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query_by_path_prefix(["parser.cpp"], top_k=5)
        assert results
        assert results[0]["file_path"] == "parser.cpp"


class TestFileHintQuery:
    """Filename hints should map to chunks even without directory paths."""

    def test_query_by_file_hints_basename(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query_by_file_hints(["parser.cpp"], top_k=5)
        assert results
        assert results[0]["file_path"] == "parser.cpp"

    def test_query_by_file_hints_stem(self):
        idx = BM25Index()
        idx.build(CHUNKS)
        results = idx.query_by_file_hints(["parser"], top_k=5)
        assert results
        assert results[0]["file_path"] == "parser.cpp"
