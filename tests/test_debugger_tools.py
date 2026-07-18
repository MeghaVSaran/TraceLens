"""Tests for bounded native-debugger evidence collection."""

from pathlib import Path
import subprocess

from src.analysis.debugger_tools import (
    build_gdb_core_command,
    collect_gdb_core_evidence,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "app"
    core = tmp_path / "core.123"
    binary.write_bytes(b"binary")
    core.write_bytes(b"core")
    return binary, core


def test_build_gdb_core_command_is_noninteractive(tmp_path):
    binary, core = _inputs(tmp_path)

    command = build_gdb_core_command("/usr/bin/gdb", binary, core)

    assert command[:4] == ["/usr/bin/gdb", "--batch", "--quiet", "--nx"]
    assert "thread apply all bt" in command
    assert "info registers" in command
    assert command[-2:] == [str(binary), str(core)]


def test_collect_gdb_reports_unavailable_without_running(tmp_path, monkeypatch):
    binary, core = _inputs(tmp_path)
    monkeypatch.setattr("src.analysis.debugger_tools.shutil.which", lambda _name: None)

    evidence = collect_gdb_core_evidence(binary, core)

    assert evidence.status == "unavailable"
    assert evidence.available is False
    assert evidence.command == []


def test_collect_gdb_captures_success_and_uses_no_shell(tmp_path, monkeypatch):
    binary, core = _inputs(tmp_path)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="#0  Engine::run () at src/engine.cc:15\n",
            stderr="",
        )

    monkeypatch.setattr("src.analysis.debugger_tools.shutil.which", lambda _name: "/usr/bin/gdb")
    monkeypatch.setattr("src.analysis.debugger_tools.subprocess.run", fake_run)

    evidence = collect_gdb_core_evidence(binary, core)

    assert evidence.status == "ok"
    assert evidence.return_code == 0
    assert "Engine::run" in evidence.log_text()
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["timeout"] == 20.0


def test_collect_gdb_returns_bounded_timeout_evidence(tmp_path, monkeypatch):
    binary, core = _inputs(tmp_path)

    def fake_run(_command, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd="gdb",
            timeout=1,
            output="x" * 200,
            stderr="still running",
        )

    monkeypatch.setattr("src.analysis.debugger_tools.shutil.which", lambda _name: "/usr/bin/gdb")
    monkeypatch.setattr("src.analysis.debugger_tools.subprocess.run", fake_run)

    evidence = collect_gdb_core_evidence(
        binary,
        core,
        timeout_seconds=1,
        max_output_chars=80,
    )

    assert evidence.status == "timeout"
    assert evidence.truncated is True
    assert len(evidence.stdout) <= 80


def test_collect_gdb_rejects_missing_inputs_before_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analysis.debugger_tools.shutil.which", lambda _name: "/usr/bin/gdb")

    evidence = collect_gdb_core_evidence(
        tmp_path / "missing-app",
        tmp_path / "missing-core",
    )

    assert evidence.status == "invalid_input"
    assert "missing-app" in evidence.stderr
