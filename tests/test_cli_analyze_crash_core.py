"""CLI tests for GDB core-dump crash evidence."""

import json

from click.testing import CliRunner

from src.analysis.debugger_tools import ToolEvidence
from src.cli.main import cli
from src.ingestion.log_parser import parse_log


def _repo_and_inputs(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".debugaid").mkdir(parents=True)
    binary = tmp_path / "app"
    core = tmp_path / "core.123"
    binary.write_bytes(b"binary")
    core.write_bytes(b"core")
    return repo, binary, core


def test_analyze_crash_requires_log_or_binary_core_pair(tmp_path):
    repo, _binary, _core = _repo_and_inputs(tmp_path)

    result = CliRunner().invoke(cli, ["analyze-crash", "--repo", str(repo)])

    assert result.exit_code != 0
    assert "Provide --log or both --binary and --core" in result.output


def test_analyze_crash_collects_gdb_core_evidence(tmp_path, monkeypatch):
    repo, binary, core = _repo_and_inputs(tmp_path)
    gdb_output = (
        "Program received signal SIGSEGV, Segmentation fault.\n"
        "#0 0x123 in Engine::run () at src/engine.cc:15\n"
        "#1 0x124 in main () at src/main.cc:5\n"
    )
    collected = ToolEvidence(
        tool="gdb",
        status="ok",
        available=True,
        command=["gdb", "--batch"],
        return_code=0,
        stdout=gdb_output,
        duration_ms=7,
    )
    observed = {}

    def fake_collect(binary_path, core_path, **kwargs):
        observed["binary"] = binary_path
        observed["core"] = core_path
        observed["kwargs"] = kwargs
        return collected

    def fake_triage(repo_path, log_text, top_k, build_dir=None, diagnose=False):
        observed["log_text"] = log_text
        return parse_log(log_text, repo_root=repo_path), [], None, None, []

    monkeypatch.setattr(
        "src.analysis.debugger_tools.collect_gdb_core_evidence",
        fake_collect,
    )
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
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    analysis = payload["crash_analysis"]
    assert analysis["crash_type"] == "segfault"
    assert analysis["crash_frame"]["function"] == "Engine::run"
    assert analysis["tool_evidence"][0]["status"] == "ok"
    assert observed["binary"] == binary
    assert observed["core"] == core
    assert "Engine::run" in observed["log_text"]


def test_analyze_crash_reports_unavailable_gdb_without_log(tmp_path, monkeypatch):
    repo, binary, core = _repo_and_inputs(tmp_path)

    monkeypatch.setattr(
        "src.analysis.debugger_tools.collect_gdb_core_evidence",
        lambda *_args, **_kwargs: ToolEvidence(
            tool="gdb",
            status="unavailable",
            available=False,
            command=[],
            stderr="GDB was not found on PATH.",
        ),
    )

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

    assert result.exit_code != 0
    assert "GDB evidence collection failed" in result.output
    assert "GDB was not found on PATH" in result.output
