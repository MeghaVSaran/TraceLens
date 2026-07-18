"""Tests for strict pathless behavior in src/evaluation/run_ablation.py."""

from dataclasses import dataclass

from src.evaluation.run_ablation import (
    SampleResult,
    _collect_ground_truth_coverage,
    _count_test_ground_truth_samples,
    _has_all_ground_truth_files,
    _is_test_like_path,
    _to_strict_pathless_parsed,
    build_ablation_report,
    run_single_config,
)
from src.ingestion.log_parser import parse_log


@dataclass
class _FakeRetrievalResult:
    rank: int
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    score: float
    dense_score: float
    bm25_score: float
    symbol_score: float = 0.0


class _CaptureRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, log_embedding, log_text, **kwargs):
        self.calls.append(
            {
                "embedding_dim": len(log_embedding),
                "log_text": log_text,
                "kwargs": kwargs,
            }
        )
        return [
            _FakeRetrievalResult(
                rank=1,
                chunk_id="absl/base/foo.cc::foo",
                file_path="absl/base/foo.cc",
                function_name="foo",
                start_line=1,
                score=0.9,
                dense_score=0.9,
                bm25_score=0.0,
                symbol_score=0.0,
            )
        ]


class _FakeParserMod:
    @staticmethod
    def parse_log(log_text, repo_root=None):
        return parse_log(log_text, repo_root=repo_root)


class _FakeLogEmbedder:
    @staticmethod
    def embed_log(parsed_log):
        return [0.0] * 768


def test_to_strict_pathless_parsed_strips_path_hints():
    parsed = parse_log(
        "/tmp/abseil/absl/strings/str_cat.cc:143:16: error: use of undeclared identifier 'foo'\n"
        "fatal error: parser/resolve.h: No such file or directory\n"
    )
    strict = _to_strict_pathless_parsed(parsed)

    assert strict.source_paths == []
    assert all("/" not in hint and "\\" not in hint for hint in strict.file_hints)
    assert strict.file_hints == []
    assert "str_cat.cc" not in strict.query_text()
    assert "/tmp/abseil/" not in strict.error_message


def test_strict_pathless_removes_paths_with_spaces_and_filename_tokens():
    parsed = parse_log(
        "/Users/dev/opencv-4.12.0 2/modules/core/src/convert.cpp:10:2: error: bad conversion\n"
        "fatal error: include/missing.hpp: No such file or directory\n"
    )

    strict = _to_strict_pathless_parsed(parsed)

    assert "convert.cpp" not in strict.raw_log
    assert "missing.hpp" not in strict.raw_log
    assert strict.file_hints == []


def test_run_single_config_passes_strict_pathless_to_retriever():
    dataset = [
        {
            "id": "s1",
            "log": "/tmp/abseil/absl/base/foo.cc:10:1: error: use of undeclared identifier 'foo'",
            "relevant_files": ["absl/base/foo.cc"],
            "error_type": "compiler_error",
        }
    ]
    config = {"name": "hybrid_with_path_boost", "mode": "hybrid", "path_boost": True}
    retriever = _CaptureRetriever()

    results = run_single_config(
        config=config,
        dataset=dataset,
        retriever=retriever,
        log_parser_mod=_FakeParserMod,
        log_embedder=_FakeLogEmbedder,
        strict_pathless=True,
    )

    assert len(results) == 1
    assert results[0].strict_pathless is True
    call = retriever.calls[0]
    assert call["kwargs"]["strict_pathless"] is True
    assert call["kwargs"]["source_paths"] == []
    assert "/tmp/abseil/" not in call["log_text"]


def test_build_report_contains_strict_no_path_bucket():
    all_results = {
        "hybrid_no_path_boost": [
            SampleResult(
                sample_id="a",
                config_name="hybrid_no_path_boost",
                error_type="linker_error",
                has_source_path=True,
                ground_truth_files=["absl/base/a.cc"],
                retrieved_files=["absl/base/a.cc"],
                ground_truth_rank=1,
                recall_at_1=1.0,
                recall_at_3=1.0,
                recall_at_5=1.0,
                mrr=1.0,
                path_boost_enabled=False,
                strict_pathless=True,
            ),
            SampleResult(
                sample_id="b",
                config_name="hybrid_no_path_boost",
                error_type="linker_error",
                has_source_path=False,
                ground_truth_files=["absl/base/b.cc"],
                retrieved_files=["absl/base/x.cc"],
                ground_truth_rank=None,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                mrr=0.0,
                path_boost_enabled=False,
                strict_pathless=False,
            ),
        ]
    }

    report = build_ablation_report(all_results)
    strict_bucket = report["by_source_path"]["strict_no_path"]["hybrid_no_path_boost"]
    assert strict_bucket["n"] == 1
    assert strict_bucket["recall_at_5"] == 1.0


def test_ground_truth_coverage_helpers(tmp_path):
    (tmp_path / "absl" / "base").mkdir(parents=True)
    (tmp_path / "absl" / "base" / "ok.cc").touch()

    present = {"relevant_files": ["absl/base/ok.cc"]}
    missing = {"relevant_files": ["absl/base/missing.cc"]}

    assert _has_all_ground_truth_files(present, tmp_path) is True
    assert _has_all_ground_truth_files(missing, tmp_path) is False

    in_repo, missing_paths = _collect_ground_truth_coverage([present, missing], tmp_path)
    assert in_repo == 1
    assert missing_paths["absl/base/missing.cc"] == 1


def test_test_ground_truth_sample_counter():
    dataset = [
        {"relevant_files": ["absl/base/internal/raw_logging.cc"]},
        {"relevant_files": ["absl/synchronization/mutex_test.cc", "absl/synchronization/mutex.h"]},
        {"relevant_files": ["absl/container/benchmarks/raw_hash_set_benchmark.cc"]},
    ]

    assert _is_test_like_path("absl/synchronization/mutex_test.cc") is True
    assert _is_test_like_path("absl/base/internal/raw_logging.cc") is False
    assert _count_test_ground_truth_samples(dataset) == 2
