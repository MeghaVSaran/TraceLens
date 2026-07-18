"""
Tests for src/indexing/vector_index.py

Uses small fake embeddings (random numpy arrays) to test build, query,
and exists without real ML models.
"""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from src.indexing.vector_index import VectorIndex, IndexNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeChunk:
    """Minimal Chunk-like object for testing (avoids importing code_parser)."""
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int


def _make_chunks_and_embeddings(n: int, dim: int = 768):
    """Create *n* fake chunks with random embeddings."""
    chunks = [
        FakeChunk(
            chunk_id=f"file_{i}.cpp::func_{i}",
            file_path=f"file_{i}.cpp",
            function_name=f"func_{i}",
            start_line=i * 10,
        )
        for i in range(n)
    ]
    rng = np.random.default_rng(seed=42)
    embeddings = [rng.random(dim).astype(np.float32) for _ in range(n)]
    return chunks, embeddings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuild:
    def test_build_creates_collection(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(5)
        idx.build(chunks, embeddings)
        assert idx.exists()

    def test_build_overwrite(self, tmp_path):
        """Building twice should overwrite, not duplicate."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(3)
        idx.build(chunks, embeddings)

        # Build again with 5 items.
        chunks2, embeddings2 = _make_chunks_and_embeddings(5)
        idx.build(chunks2, embeddings2)

        results = idx.query(embeddings2[0].tolist(), top_k=10)
        assert len(results) == 5


    def test_add_supports_multiple_batches(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(4)
        idx.reset()
        idx.add(chunks[:2], embeddings[:2])
        idx.add(chunks[2:], embeddings[2:])
        assert len(idx.query(embeddings[0], top_k=10)) == 4


class TestQuery:
    def test_returns_correct_fields(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(5)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0].tolist(), top_k=3)
        assert len(results) == 3

        r = results[0]
        assert "chunk_id" in r
        assert "file_path" in r
        assert "function_name" in r
        assert "start_line" in r
        assert "score" in r

    def test_top_result_is_self(self, tmp_path):
        """Querying with an existing embedding should return it as top-1."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(5)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0].tolist(), top_k=1)
        assert results[0]["chunk_id"] == chunks[0].chunk_id

    def test_score_range(self, tmp_path):
        """Cosine similarity scores should be in roughly [0, 1]."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(10)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0].tolist(), top_k=10)
        for r in results:
            assert -0.1 <= r["score"] <= 1.1  # small tolerance

    def test_top_k_limits_results(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(10)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0].tolist(), top_k=3)
        assert len(results) == 3

    def test_query_accepts_numpy(self, tmp_path):
        """query() should accept numpy arrays, not just lists."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(5)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0], top_k=2)  # numpy, not .tolist()
        assert len(results) == 2


class TestExists:
    def test_false_before_build(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        assert idx.exists() is False

    def test_true_after_build(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(3)
        idx.build(chunks, embeddings)
        assert idx.exists() is True


class TestIndexNotFoundError:
    def test_query_before_build_raises(self, tmp_path):
        idx = VectorIndex(tmp_path / "chroma")
        rng = np.random.default_rng(seed=0)
        fake_query = rng.random(768).astype(np.float32).tolist()

        with pytest.raises(IndexNotFoundError):
            idx.query(fake_query)


class TestPersistence:
    def test_survives_restart(self, tmp_path):
        """Index should persist across VectorIndex instances."""
        persist_dir = tmp_path / "chroma"
        chunks, embeddings = _make_chunks_and_embeddings(5)

        # Build with first instance.
        idx1 = VectorIndex(persist_dir)
        idx1.build(chunks, embeddings)

        # Create new instance pointing at same directory.
        idx2 = VectorIndex(persist_dir)
        assert idx2.exists()

        results = idx2.query(embeddings[0].tolist(), top_k=1)
        assert results[0]["chunk_id"] == chunks[0].chunk_id


class TestWhereFilter:
    """Test the optional ``where`` parameter for metadata filtering."""

    def test_where_filters_by_file_path(self, tmp_path):
        """Querying with where={file_path: {$eq: X}} returns only chunks from X."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(10)
        idx.build(chunks, embeddings)

        target_file = chunks[3].file_path  # "file_3.cpp"
        results = idx.query(
            embeddings[0].tolist(),
            top_k=10,
            where={"file_path": {"$eq": target_file}},
        )
        # All returned results should be from the target file
        for r in results:
            assert r["file_path"] == target_file

    def test_where_none_returns_all(self, tmp_path):
        """Querying without where returns results from any file."""
        idx = VectorIndex(tmp_path / "chroma")
        chunks, embeddings = _make_chunks_and_embeddings(5)
        idx.build(chunks, embeddings)

        results = idx.query(embeddings[0].tolist(), top_k=5, where=None)
        assert len(results) == 5
