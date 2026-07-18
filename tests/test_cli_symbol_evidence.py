"""CLI coverage for native symbol evidence feeding crash retrieval."""

from click.testing import CliRunner

from src.analysis.debugger_tools import ToolEvidence
from src.analysis.symbol_tools import SymbolToolEvidence
from src.cli.main import cli
from src.ingestion.log_parser import parse_log


def test_analyze_crash_adds_symbol_records_to_retrieval_context(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".debugaid").mkdir(parents=True)
    binary = tmp_path / "app"
    core = tmp_path / "core.123"
    binary.write_bytes(b"binary")
    core.write_bytes(b"core")
    observed = {}

    gdb = ToolEvidence(
        tool="gdb",
        status="ok",
        available=True,
        command=["gdb"],
        return_code=0,
        stdout=(
            "Program received signal SIGSEGV, Segmentation fault.\n"
            "#0 0x401123 in Engine::run () at src/engine.cc:15\n"
        ),
    )
    addr2line = SymbolToolEvidence(
        tool="addr2line",
        status="ok",
        available=True,
        command=["addr2line"],
        return_code=0,
        duration_ms=3,
        version="GNU addr2line 2.42",
        records=[
            {
                "address": "0x401123",
                "function": "Engine::run()",
                "file_path": "src/engine.cc",
                "line_number": 15,
                "display": "0x401123 -> Engine::run() at src/engine.cc:15",
            }
        ],
    )

    monkeypatch.setattr(
        "src.analysis.debugger_tools.collect_gdb_core_evidence",
        lambda *_args, **_kwargs: gdb,
    )
    monkeypatch.setattr(
        "src.analysis.symbol_tools.collect_native_symbol_evidence",
        lambda *_args, **_kwargs: [addr2line],
    )

    def fake_triage(repo_path, log_text, top_k, build_dir=None, diagnose=False):
        observed["log_text"] = log_text
        return parse_log(log_text, repo_root=repo_path), [], None, None, []

    monkeypatch.setattr("src.cli.main._triage_log", fake_triage)

    result = CliRunner().invoke(
        cli,
        [
            "analyze-crash",
            "--repo",
            str(repo),
            "--binary",
            str(binary),
            "--core",
            str(core),
        ],
    )

    assert result.exit_code == 0
    assert "addr2line: status=ok" in result.output
    assert "GNU addr2line 2.42" in result.output
    assert "0x401123 -> Engine::run() at src/engine.cc:15" in result.output
    assert "Native symbol evidence:" in observed["log_text"]
    assert "Engine::run()" in observed["log_text"]
