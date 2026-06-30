"""Tests for runtime crash triage analysis."""

from dataclasses import dataclass

from src.analysis.crash_analyzer import analyze_crash
from src.ingestion.log_parser import parse_log


@dataclass
class _Result:
    rank: int = 1
    chunk_id: str = "1"
    file_path: str = "src/token_stream.cc"
    function_name: str = "TokenStream::next"
    start_line: int = 42
    score: float = 0.91
    dense_score: float = 0.4
    bm25_score: float = 0.8
    symbol_score: float = 0.3


ASAN_USE_AFTER_FREE = """
==123==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010
READ of size 4 at 0x602000000010 thread T0
#0 0x555555 in TokenStream::next src/token_stream.cc:42
#1 0x555556 in Parser::parse src/parser.cc:88
#2 0x555557 in main src/main.cc:12
freed by thread T0 here:
#0 0x7ffff in operator delete(void*) asan_new_delete.cpp:152
#1 0x555558 in Parser::reset src/parser.cc:70
"""


def test_analyze_crash_classifies_asan_and_frames(tmp_path):
    source = tmp_path / "src" / "token_stream.cc"
    source.parent.mkdir(parents=True)
    source.write_text("int TokenStream::next() { return 0; }\n", encoding="utf-8")
    parsed = parse_log(ASAN_USE_AFTER_FREE)

    report = analyze_crash(parsed, [_Result()], repo_root=tmp_path)

    assert report.crash_type == "heap-use-after-free"
    assert report.crash_frame is not None
    assert report.crash_frame.function == "TokenStream::next"
    assert "accessed after its lifetime ended" in report.likely_cause
    assert report.confidence == "high"
    assert any("src/token_stream.cc" in match for match in report.exact_matches)


def test_analyze_crash_handles_plain_gdb_backtrace():
    log = """
Program received signal SIGSEGV, Segmentation fault.
#0  0x00005555555551a9 in Engine::run (this=0x0) at src/engine.cc:15
#1  0x0000555555555230 in main () at src/main.cc:5
"""
    parsed = parse_log(log)

    report = analyze_crash(parsed, [_Result(file_path="src/engine.cc", function_name="Engine::run")])

    assert report.crash_type == "segfault"
    assert report.crash_frame is not None
    assert report.crash_frame.file_path == "src/engine.cc"
    assert "invalid pointer" in report.likely_cause
