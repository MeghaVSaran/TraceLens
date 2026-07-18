"""Tests for runtime crash triage analysis."""

from dataclasses import dataclass

from src.analysis.crash_analyzer import analyze_crash
from src.ingestion.log_parser import parse_log


@dataclass
class _Result:
    rank: int = 1
    chunk_id: str = "1"
    file_path: str = "src/parser/token_stream.cc"
    function_name: str = "TokenStream::next"
    start_line: int = 42
    score: float = 0.91
    dense_score: float = 0.4
    bm25_score: float = 0.8
    symbol_score: float = 0.3


ASAN_USE_AFTER_FREE = """
==123==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010
READ of size 4 at 0x602000000010 thread T0
#0 0x555555 in TokenStream::next src/parser/token_stream.cc:42
#1 0x555556 in Parser::next src/parser/parser.cc:88
#2 0x555557 in main src/main.cc:12
freed by thread T0 here:
#0 0x555558 in Session::clear src/parser/session.cc:120
#1 0x555559 in Parser::reset src/parser/parser.cc:70
previously allocated by thread T0 here:
#0 0x555560 in TokenStream::TokenStream src/parser/token_stream.cc:15
#1 0x555561 in Parser::Parser src/parser/parser.cc:20
"""


def test_asan_use_after_free_parses_free_and_allocation_stacks(tmp_path):
    parsed = parse_log(ASAN_USE_AFTER_FREE)

    report = analyze_crash(parsed, [_Result()], repo_root=tmp_path)

    assert report.crash_type == "heap-use-after-free"
    assert report.crash_frame is not None
    assert report.crash_frame.function == "TokenStream::next"
    assert [frame.function for frame in report.freed_by_frames] == ["Session::clear", "Parser::reset"]
    assert [frame.function for frame in report.allocated_by_frames] == [
        "TokenStream::TokenStream",
        "Parser::Parser",
    ]


def test_ownership_keyword_scoring_ranks_free_frame_above_crash_caller(tmp_path):
    parsed = parse_log(ASAN_USE_AFTER_FREE)

    report = analyze_crash(parsed, [_Result()], repo_root=tmp_path)

    top = report.suspicious_frames[0]
    parser_next = next(frame for frame in report.suspicious_frames if frame.function == "Parser::next")
    assert top.function == "Session::clear"
    assert top.score > parser_next.score
    assert top.score >= 1.0


def test_confidence_high_when_free_stack_has_ownership_keyword(tmp_path):
    parsed = parse_log(ASAN_USE_AFTER_FREE)

    report = analyze_crash(parsed, [_Result()], repo_root=tmp_path)

    assert report.confidence == "high"
    assert "Check Session::clear - this is where the object was freed" in report.suggested_checks
    assert any("Inspect object lifetime of TokenStream after Session::clear" == check for check in report.suggested_checks)


def test_confidence_low_when_only_crash_frame_available():
    log = """
Program received signal SIGSEGV, Segmentation fault.
#0  0x00005555555551a9 in Engine::run (this=0x0) at src/engine.cc:15
"""
    parsed = parse_log(log)

    report = analyze_crash(parsed, [])

    assert report.crash_type == "segfault"
    assert report.confidence == "low"
    assert len(report.suspicious_frames) == 1
    assert "Compile with -fsanitize=address for more precise information" in report.suggested_checks


def test_rg_falls_back_to_python_search_when_rg_missing(tmp_path, monkeypatch):
    source = tmp_path / "src" / "parser" / "session.cc"
    source.parent.mkdir(parents=True)
    source.write_text("void Session::clear() { state_.reset(); }\n", encoding="utf-8")
    parsed = parse_log(ASAN_USE_AFTER_FREE)
    monkeypatch.setattr("src.analysis.crash_analyzer.shutil.which", lambda _name: None)

    report = analyze_crash(parsed, [_Result()], repo_root=tmp_path)

    assert report.rg_evidence["Session::clear"] == ["src/parser/session.cc"]


def test_structured_output_contains_required_keys(tmp_path):
    parsed = parse_log(ASAN_USE_AFTER_FREE)

    data = analyze_crash(parsed, [_Result()], repo_root=tmp_path).to_dict()

    assert set([
        "crash_type",
        "crash_frame",
        "freed_by_frames",
        "allocated_by_frames",
        "suspicious_frames",
        "rg_evidence",
        "confidence",
        "suggested_checks",
    ]).issubset(data.keys())
    assert "score" in data["suspicious_frames"][0]
