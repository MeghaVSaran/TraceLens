"""
Tests for src/retrieval/hybrid_retriever.py

Uses simple fake index classes (no real ChromaDB or BM25 needed)
to verify fusion math, fallback behaviour, and sort order.
"""

import pytest
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    SOURCE_PATH_SCORE,
)
from src.indexing.vector_index import IndexNotFoundError
from src.ingestion.log_parser import parse_log


# ---------------------------------------------------------------------------
# Fake index classes
# ---------------------------------------------------------------------------

class FakeVectorIndex:
    """Returns canned results for query().

    Supports optional ``where`` filter: when ``where`` is provided and
    ``filtered_results`` was set at init, those are returned instead.
    """

    def __init__(self, results=None, raise_error=False, filtered_results=None):
        self._results = results or []
        self._raise_error = raise_error
        self._filtered_results = filtered_results or []

    def query(self, log_embedding, top_k=20, where=None):
        if self._raise_error:
            raise IndexNotFoundError("No collection")
        if where is not None:
            return self._filtered_results[:top_k]
        return self._results[:top_k]


class FakeBM25Index:
    """Returns canned results for query()."""

    def __init__(self, results=None, symbol_results=None):
        self._results = results or []
        self._symbol_results = symbol_results or []

    def query(self, text, top_k=20):
        return self._results

    def query_by_symbols(self, identifiers, top_k=20):
        return self._symbol_results[:top_k]

    def query_by_path_prefix(self, prefixes, top_k=20):
        # Reuse symbol results for simple fake-path-prefix behavior in tests.
        return self._symbol_results[:top_k]

    def query_by_file_hints(self, hints, top_k=20):
        # Reuse symbol results for simple fake-file-hint behavior in tests.
        return self._symbol_results[:top_k]


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DENSE_RESULTS = [
    {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
     "function_name": "foo", "symbol_name": "foo", "signature": "void foo()",
     "start_line": 10, "score": 0.9},
    {"chunk_id": "b.cpp::bar", "file_path": "b.cpp",
     "function_name": "bar", "symbol_name": "bar", "signature": "void bar()",
     "start_line": 20, "score": 0.7},
    {"chunk_id": "c.cpp::baz", "file_path": "c.cpp",
     "function_name": "baz", "symbol_name": "baz", "signature": "void baz()",
     "start_line": 30, "score": 0.5},
]

BM25_RESULTS = [
    {"chunk_id": "b.cpp::bar", "file_path": "b.cpp",
     "function_name": "bar", "symbol_name": "bar", "signature": "void bar()",
     "start_line": 20, "score": 8.0},
    {"chunk_id": "d.cpp::qux", "file_path": "d.cpp",
     "function_name": "qux", "symbol_name": "qux", "signature": "void qux()",
     "start_line": 40, "score": 5.0},
    {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
     "function_name": "foo", "symbol_name": "foo", "signature": "void foo()",
     "start_line": 10, "score": 2.0},
]

FAKE_EMBEDDING = [0.0] * 768


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFusionMath:
    """Verify that score normalization and fusion weights are correct."""

    def test_scores_in_unit_range(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-6

    def test_dense_and_bm25_scores_normalized(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            assert 0.0 <= r.dense_score <= 1.0 + 1e-6
            assert 0.0 <= r.bm25_score <= 1.0 + 1e-6

    def test_fusion_weights(self):
        """Fused score should equal 0.5 * dense + 0.5 * bm25."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            expected = DENSE_WEIGHT * r.dense_score + SPARSE_WEIGHT * r.bm25_score
            assert abs(r.score - expected) < 1e-6

    def test_overlapping_chunk_fused(self):
        """Chunk b.cpp::bar appears in both; should have non-zero dense and bm25."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        bar = [r for r in results if r.chunk_id == "b.cpp::bar"][0]
        assert bar.dense_score > 0.0
        assert bar.bm25_score > 0.0

    def test_unique_chunk_has_zero_for_missing_index(self):
        """d.cpp::qux only in BM25 → dense_score should be 0."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        qux = [r for r in results if r.chunk_id == "d.cpp::qux"][0]
        assert qux.dense_score == 0.0
        assert qux.bm25_score > 0.0


class TestSorting:
    """Results should be sorted by fused score descending with correct ranks."""

    def test_descending_order(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_top_k_limits(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=2)
        assert len(results) == 2


class TestFallback:
    """Graceful degradation when one index is unavailable."""

    def test_dense_fails_falls_back_to_bm25(self):
        retriever = HybridRetriever(
            FakeVectorIndex(raise_error=True),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert len(results) > 0
        # All dense_scores should be 0 (only BM25 contributed).
        for r in results:
            assert r.dense_score == 0.0
        # At least the top result should have a non-zero bm25_score;
        # the lowest-scoring item normalizes to 0.0 (min-max).
        assert results[0].bm25_score > 0.0

    def test_bm25_empty_falls_back_to_dense(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.bm25_score == 0.0
        # At least the top result should have a non-zero dense_score.
        assert results[0].dense_score > 0.0

    def test_both_empty_returns_empty(self):
        retriever = HybridRetriever(
            FakeVectorIndex([]),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert results == []


class TestRetrievalResult:
    """Verify the returned objects are proper RetrievalResult instances."""

    def test_returns_retrieval_results(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=3)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_total_unique_chunks(self):
        """Union of 3 dense + 3 BM25 results (2 overlap) = 4 unique."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        # With dedup on, each file only appears once — all 4 have unique file_paths
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10, deduplicate_files=False)
        assert len(results) == 4


class TestFileDeduplication:
    """Fix 5: File-path deduplication keeps only the best result per file."""

    # Two functions from the same file
    DENSE_SAME_FILE = [
        {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
         "function_name": "foo", "start_line": 10, "score": 0.9},
        {"chunk_id": "a.cpp::bar", "file_path": "a.cpp",
         "function_name": "bar", "start_line": 50, "score": 0.8},
        {"chunk_id": "b.cpp::baz", "file_path": "b.cpp",
         "function_name": "baz", "start_line": 30, "score": 0.5},
    ]

    def test_dedup_removes_same_file_duplicates(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=True)
        file_paths = [r.file_path for r in results]
        assert len(file_paths) == len(set(file_paths)), "Duplicate file_paths in results!"
        assert len(results) == 2  # a.cpp and b.cpp

    def test_dedup_keeps_highest_score(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=True)
        a_result = [r for r in results if r.file_path == "a.cpp"][0]
        assert a_result.function_name == "foo"  # foo has higher score (0.9 > 0.8)

    def test_dedup_off_keeps_all(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=False)
        assert len(results) == 3  # All three results kept


class TestSourcePathInjection:
    """Fix 4: Direct source path chunk injection from ChromaDB."""

    # Chunks that would be returned by the where-filter query
    INJECTED_CHUNKS = [
        {"chunk_id": "target.cc::func_x", "file_path": "target.cc",
         "function_name": "func_x", "start_line": 23, "score": 0.3},
    ]

    def test_injection_adds_missing_file(self):
        """A file not in hybrid results should appear after injection."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=False,
        )
        file_paths = [r.file_path for r in results]
        assert "target.cc" in file_paths

    def test_injected_chunk_has_high_score(self):
        """Injected chunks should get SOURCE_PATH_SCORE (0.95)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=False,
        )
        target = [r for r in results if r.file_path == "target.cc"][0]
        assert target.score == SOURCE_PATH_SCORE

    def test_injected_chunk_ranks_first(self):
        """Injected file should be rank 1 (highest score)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=True,
        )
        assert results[0].file_path == "target.cc"
        assert results[0].rank == 1

    def test_no_source_paths_no_injection(self):
        """Without source_paths, no injection occurs."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=None, deduplicate_files=False,
        )
        file_paths = [r.file_path for r in results]
        assert "target.cc" not in file_paths

    def test_injection_upgrades_existing_chunk(self):
        """If injected chunk already in fused pool, its score is upgraded."""
        # a.cpp is already in DENSE_RESULTS
        existing_chunk = [
            {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
             "function_name": "foo", "start_line": 10, "score": 0.2},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=existing_chunk),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["a.cpp"], deduplicate_files=False,
        )
        a_result = [r for r in results if r.file_path == "a.cpp"][0]
        # Score should be upgraded to SOURCE_PATH_SCORE
        assert a_result.score == SOURCE_PATH_SCORE


# ---------------------------------------------------------------------------
# 8. Retrieval mode switching
# ---------------------------------------------------------------------------

class TestRetrievalModes:
    """Tests for the mode and path_boost parameters added for ablation."""

    def test_bm25_only_mode_ignores_dense(self):
        """In bm25 mode, dense index is not queried at all."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=5,
            mode="bm25",
        )
        # All results should have dense_score = 0
        for r in results:
            assert r.dense_score == 0.0
        # BM25 results should be present
        assert len(results) > 0

    def test_dense_only_mode_ignores_bm25(self):
        """In dense mode, BM25 is not queried."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=5,
            mode="dense",
        )
        # All results should have bm25_score = 0
        for r in results:
            assert r.bm25_score == 0.0
        assert len(results) > 0

    def test_path_boost_false_suppresses_injection(self):
        """When path_boost=False, source_paths are ignored."""
        filtered = [
            {"chunk_id": "x.cpp::xfn", "file_path": "x.cpp",
             "function_name": "xfn", "start_line": 1, "score": 0.99},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=filtered),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=5,
            source_paths=["x.cpp"],
            path_boost=False,
        )
        # x.cpp should NOT be in results since path boost is disabled
        file_paths = [r.file_path for r in results]
        assert "x.cpp" not in file_paths

    def test_path_boost_true_injects(self):
        """When path_boost=True, source_paths chunks are injected."""
        filtered = [
            {"chunk_id": "x.cpp::xfn", "file_path": "x.cpp",
             "function_name": "xfn", "start_line": 1, "score": 0.99},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=filtered),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=5,
            source_paths=["x.cpp"],
            path_boost=True,
        )
        file_paths = [r.file_path for r in results]
        assert "x.cpp" in file_paths

    def test_bm25_with_path_boost(self):
        """BM25 mode with path_boost enabled also injects source paths."""
        filtered = [
            {"chunk_id": "x.cpp::xfn", "file_path": "x.cpp",
             "function_name": "xfn", "start_line": 1, "score": 0.99},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex([], filtered_results=filtered),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=5,
            source_paths=["x.cpp"],
            mode="bm25",
            path_boost=True,
        )
        file_paths = [r.file_path for r in results]
        assert "x.cpp" in file_paths

    def test_default_mode_is_hybrid(self):
        """Without explicit mode, retriever uses hybrid (both indices)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=5)
        # b.cpp::bar appears in both indices so should have both scores > 0
        bar = [r for r in results if r.function_name == "bar"]
        assert len(bar) == 1
        assert bar[0].dense_score > 0.0
        assert bar[0].bm25_score > 0.0


class TestSymbolAwareBoosting:
    """Symbol-aware ranking should help pathless symbol queries."""

    SYMBOL_DENSE = [
        {"chunk_id": "generic.cpp::helper", "file_path": "generic.cpp",
         "function_name": "helper", "symbol_name": "helper",
         "signature": "void helper()", "start_line": 8, "score": 0.95},
        {"chunk_id": "parser.cpp::Parser::resolveSymbol", "file_path": "parser.cpp",
         "function_name": "Parser::resolveSymbol", "symbol_name": "resolveSymbol",
         "class_name": "Parser", "signature": "void Parser::resolveSymbol(Symbol& s)",
         "start_line": 42, "score": 0.60},
    ]

    def test_exact_identifier_match_boosts_target(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.SYMBOL_DENSE),
            FakeBM25Index([]),
        )
        parsed_log = parse_log("undefined reference to `Parser::resolveSymbol`")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed_log.query_text(),
            top_k=5,
            parsed_log=parsed_log,
            deduplicate_files=False,
        )
        assert results[0].function_name == "Parser::resolveSymbol"
        assert results[0].symbol_score > 0.0

    def test_stack_frame_match_boosts_target(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.SYMBOL_DENSE),
            FakeBM25Index([]),
        )
        parsed_log = parse_log(
            "Segmentation fault\n"
            "#0  0x1 in Parser::resolveSymbol (this=0x0) at parser.cpp:42\n"
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed_log.query_text(),
            top_k=5,
            parsed_log=parsed_log,
            deduplicate_files=False,
        )
        assert results[0].function_name == "Parser::resolveSymbol"

    def test_symbol_candidates_can_surface_missing_chunk(self):
        retriever = HybridRetriever(
            FakeVectorIndex([self.SYMBOL_DENSE[0]]),
            FakeBM25Index(
                results=[],
                symbol_results=[self.SYMBOL_DENSE[1]],
            ),
        )
        parsed_log = parse_log("undefined reference to `Parser::resolveSymbol`")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed_log.query_text(),
            top_k=5,
            parsed_log=parsed_log,
            deduplicate_files=False,
        )
        assert any(r.function_name == "Parser::resolveSymbol" for r in results)


class TestNormalizationEdgeCases:
    """Single-score normalization should still preserve signal."""

    def test_single_score_normalizes_to_one(self):
        retriever = HybridRetriever(
            FakeVectorIndex([
                {"chunk_id": "only.cpp::one", "file_path": "only.cpp",
                 "function_name": "one", "start_line": 1, "score": 0.2},
            ]),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "one", top_k=1)
        assert results[0].dense_score == pytest.approx(1.0)


class TestPathContextBoosting:
    """Path hints should influence ranking even without explicit identifiers."""

    def test_component_prefix_hint_boosts_matching_paths(self):
        dense = [
            {"chunk_id": "a.cpp::x", "file_path": "absl/base/config.h",
             "function_name": "x", "start_line": 1, "score": 0.6},
            {"chunk_id": "b.cpp::y", "file_path": "absl/strings/str_cat.cc",
             "function_name": "y", "start_line": 1, "score": 0.6},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(dense),
            FakeBM25Index([]),
        )
        parsed = parse_log("linker output")
        parsed.source_paths = ["absl/base/internal/low_level_alloc_test.cc"]
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            "linker output",
            top_k=2,
            parsed_log=parsed,
            deduplicate_files=False,
        )
        assert results[0].file_path == "absl/base/config.h"

    def test_basename_file_hint_can_surface_candidate(self):
        dense = [
            {"chunk_id": "x.cpp::x", "file_path": "noise.cpp",
             "function_name": "x", "start_line": 1, "score": 0.7},
        ]
        hinted = [
            {"chunk_id": "target.cpp::y", "file_path": "target.cpp",
             "function_name": "y", "start_line": 1, "score": 3.0},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(dense),
            FakeBM25Index(results=[], symbol_results=hinted),
        )
        parsed = parse_log("error in target.cpp")
        parsed.file_hints = ["target.cpp"]
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed.query_text(),
            top_k=2,
            parsed_log=parsed,
            deduplicate_files=False,
        )
        assert any(r.file_path == "target.cpp" for r in results)


class TestTestFilePenalty:
    """Test files should not crowd out production files unless query points to tests."""

    def test_non_test_query_downweights_test_file(self):
        dense = [
            {"chunk_id": "prod.h::Condition", "file_path": "absl/synchronization/mutex.h",
             "function_name": "Condition", "symbol_name": "Condition", "start_line": 10, "score": 0.90},
            {"chunk_id": "test.cc::TEST", "file_path": "absl/synchronization/mutex_test.cc",
             "function_name": "TEST", "symbol_name": "TEST", "start_line": 20, "score": 0.96},
        ]
        retriever = HybridRetriever(FakeVectorIndex(dense), FakeBM25Index([]))
        parsed = parse_log("no matching function for call to `absl::Condition::Condition`")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed.query_text(),
            top_k=2,
            parsed_log=parsed,
            deduplicate_files=False,
            mode="dense",
        )
        assert results[0].file_path == "absl/synchronization/mutex.h"

    def test_test_query_keeps_test_file_ranked(self):
        dense = [
            {"chunk_id": "prod.h::Condition", "file_path": "absl/synchronization/mutex.h",
             "function_name": "Condition", "symbol_name": "Condition", "start_line": 10, "score": 0.90},
            {"chunk_id": "test.cc::TEST", "file_path": "absl/synchronization/mutex_test.cc",
             "function_name": "TEST", "symbol_name": "TEST", "start_line": 20, "score": 0.96},
        ]
        retriever = HybridRetriever(FakeVectorIndex(dense), FakeBM25Index([]))
        parsed = parse_log("absl/synchronization/mutex_test.cc: error: no matching function for call")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed.query_text(),
            top_k=2,
            parsed_log=parsed,
            deduplicate_files=False,
            mode="dense",
        )
        assert results[0].file_path == "absl/synchronization/mutex_test.cc"


class TestAdaptiveFusionWeights:
    """Hybrid fusion should adapt to linker/compiler-heavy lexical queries."""

    def test_linker_query_prefers_sparse_signal(self):
        dense = [
            {"chunk_id": "good.cpp::target", "file_path": "good.cpp",
             "function_name": "target", "start_line": 1, "score": 0.2},
            {"chunk_id": "bad.cpp::noise", "file_path": "bad.cpp",
             "function_name": "noise", "start_line": 1, "score": 0.9},
        ]
        bm25 = [
            {"chunk_id": "good.cpp::target", "file_path": "good.cpp",
             "function_name": "target", "start_line": 1, "score": 10.0},
            {"chunk_id": "bad.cpp::noise", "file_path": "bad.cpp",
             "function_name": "noise", "start_line": 1, "score": 1.0},
        ]
        retriever = HybridRetriever(FakeVectorIndex(dense), FakeBM25Index(bm25))
        parsed = parse_log("undefined reference to `target`")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed.query_text(),
            top_k=2,
            parsed_log=parsed,
            deduplicate_files=False,
        )
        assert results[0].file_path == "good.cpp"

    def test_strict_pathless_linker_raises_sparse_weight(self):
        retriever = HybridRetriever(FakeVectorIndex([]), FakeBM25Index([]))
        parsed = parse_log("undefined reference to `absl::StrCat`")
        dense_normal, sparse_normal = retriever._select_fusion_weights(
            parsed,
            strict_pathless=False,
        )
        dense_strict, sparse_strict = retriever._select_fusion_weights(
            parsed,
            strict_pathless=True,
        )
        assert sparse_strict > sparse_normal
        assert dense_strict < dense_normal
        assert sparse_strict >= 0.9

    def test_strict_pathless_segfault_raises_sparse_weight(self):
        retriever = HybridRetriever(FakeVectorIndex([]), FakeBM25Index([]))
        parsed = parse_log(
            "Segmentation fault\n"
            "#0  0x1 in Parser::resolveSymbol (this=0x0) at parser.cpp:42\n"
        )
        _, sparse_normal = retriever._select_fusion_weights(parsed, strict_pathless=False)
        _, sparse_strict = retriever._select_fusion_weights(parsed, strict_pathless=True)
        assert sparse_strict > sparse_normal

    def test_sparse_anchor_favors_lexical_top_candidate(self):
        dense = [
            {"chunk_id": "wrong.cpp::noise", "file_path": "wrong.cpp",
             "function_name": "noise", "start_line": 1, "score": 0.95},
            {"chunk_id": "good.cpp::target", "file_path": "good.cpp",
             "function_name": "target", "start_line": 1, "score": 0.40},
        ]
        bm25 = [
            {"chunk_id": "good.cpp::target", "file_path": "good.cpp",
             "function_name": "target", "start_line": 1, "score": 8.0},
            {"chunk_id": "wrong.cpp::noise", "file_path": "wrong.cpp",
             "function_name": "noise", "start_line": 1, "score": 7.9},
        ]
        retriever = HybridRetriever(FakeVectorIndex(dense), FakeBM25Index(bm25))
        parsed = parse_log("undefined reference to `target`")
        results = retriever.retrieve(
            FAKE_EMBEDDING,
            parsed.query_text(),
            top_k=2,
            parsed_log=parsed,
            strict_pathless=True,
            deduplicate_files=False,
        )
        assert results[0].file_path == "good.cpp"

    def test_confidence_gate_blocks_sparse_anchor_on_weak_lexical_signal(self):
        retriever = HybridRetriever(FakeVectorIndex([]), FakeBM25Index([]))
        parsed = parse_log("Segmentation fault")
        rows = [
            RetrievalResult(
                rank=1,
                chunk_id="good.cpp::target",
                file_path="good.cpp",
                function_name="target",
                start_line=1,
                score=0.40,
                dense_score=0.40,
                bm25_score=1.00,
                dense_raw_score=0.40,
                bm25_raw_score=10.00,
                symbol_score=0.0,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="wrong.cpp::noise",
                file_path="wrong.cpp",
                function_name="noise",
                start_line=1,
                score=0.91,
                dense_score=0.91,
                bm25_score=0.99,
                dense_raw_score=0.91,
                bm25_raw_score=9.95,
                symbol_score=0.0,
            ),
        ]
        anchored = retriever._apply_sparse_anchor(
            rows,
            parsed_log=parsed,
            strict_pathless=True,
        )
        # With weak lexical confidence, gate should keep scores unchanged.
        assert anchored[0].score == pytest.approx(0.40)
        assert anchored[1].score == pytest.approx(0.91)

    def test_confidence_gate_allows_sparse_anchor_on_strong_lexical_signal(self):
        retriever = HybridRetriever(FakeVectorIndex([]), FakeBM25Index([]))
        parsed = parse_log("undefined reference to `target`")
        rows = [
            RetrievalResult(
                rank=1,
                chunk_id="good.cpp::target",
                file_path="good.cpp",
                function_name="target",
                start_line=1,
                score=0.45,
                dense_score=0.45,
                bm25_score=1.00,
                dense_raw_score=0.45,
                bm25_raw_score=12.0,
                symbol_score=0.55,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="wrong.cpp::noise",
                file_path="wrong.cpp",
                function_name="noise",
                start_line=1,
                score=0.90,
                dense_score=0.90,
                bm25_score=0.20,
                dense_raw_score=0.90,
                bm25_raw_score=4.0,
                symbol_score=0.0,
            ),
        ]
        anchored = retriever._apply_sparse_anchor(
            rows,
            parsed_log=parsed,
            strict_pathless=True,
        )
        assert anchored[0].score > 0.90

    def test_strict_top1_guard_promotes_strong_bm25_candidate(self):
        retriever = HybridRetriever(FakeVectorIndex([]), FakeBM25Index([]))
        parsed = parse_log("undefined reference to `target`")
        rows = [
            RetrievalResult(
                rank=1,
                chunk_id="wrong.cpp::noise",
                file_path="wrong.cpp",
                function_name="noise",
                start_line=1,
                score=0.92,
                dense_score=0.92,
                bm25_score=0.10,
                dense_raw_score=0.92,
                bm25_raw_score=1.0,
                symbol_score=0.0,
            ),
            RetrievalResult(
                rank=2,
                chunk_id="good.cpp::target",
                file_path="good.cpp",
                function_name="target",
                start_line=1,
                score=0.91,
                dense_score=0.20,
                bm25_score=1.00,
                dense_raw_score=0.20,
                bm25_raw_score=10.0,
                symbol_score=0.40,
            ),
        ]
        guarded = retriever._apply_strict_pathless_top1_guard(rows, parsed_log=parsed)
        assert guarded[0].chunk_id == "good.cpp::target"
