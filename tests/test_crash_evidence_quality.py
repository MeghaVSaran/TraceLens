"""Regression tests for crash-evidence scoring and provenance."""

from src.analysis.crash_analyzer import _keyword_bonus, analyze_crash
from src.analysis.debugger_tools import ToolEvidence
from src.ingestion.log_parser import parse_log


def test_short_at_keyword_does_not_match_unrelated_function_name():
    assert _keyword_bonus("TokenStream::next") == 0.0
    assert _keyword_bonus("Container::at") == 0.2


def test_camel_case_ownership_keyword_is_detected():
    assert _keyword_bonus("Session::clearCache") == 0.3


def test_tool_evidence_is_preserved_and_cited():
    parsed = parse_log(
        "Program received signal SIGSEGV\n"
        "#0 0x123 in Engine::run () at src/engine.cc:15\n"
    )
    gdb = ToolEvidence(
        tool="gdb",
        status="ok",
        available=True,
        command=["gdb", "--batch"],
        return_code=0,
    )

    report = analyze_crash(parsed, [], tool_evidence=[gdb])
    data = report.to_dict()

    assert data["tool_evidence"][0]["tool"] == "gdb"
    assert "gdb evidence collection status: ok." in report.evidence
