"""Tests for native symbolization evidence collectors."""

from pathlib import Path
import subprocess

from src.analysis.symbol_tools import (
    SymbolToolEvidence,
    collect_addr2line_evidence,
    collect_cxxfilt_evidence,
    collect_native_symbol_evidence,
    collect_nm_evidence,
    extract_frame_addresses,
    extract_frame_symbols,
    extract_mangled_symbols,
    symbol_records_text,
)


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "app"
    binary.write_bytes(b"binary")
    return binary


def _fake_version(command, **_kwargs):
    return subprocess.CompletedProcess(command, 0, stdout="GNU tool 2.42\n", stderr="")


def test_extracts_unique_frame_addresses_only():
    text = (
        "fault address 0xdeadbeef\n"
        "#0 0x401123 in Engine::run () at src/engine.cc:15\n"
        "#1 0x401200 in main () at src/main.cc:5\n"
        "#2 0x401123 in Engine::run () at src/engine.cc:15\n"
    )

    assert extract_frame_addresses(text) == ["0x401123", "0x401200"]


def test_extracts_unique_mangled_symbols():
    text = "undefined reference to `_ZN6Engine3runEPi`; repeated _ZN6Engine3runEPi"

    assert extract_mangled_symbols(text) == ["_ZN6Engine3runEPi"]


def test_extracts_frame_symbols_without_arguments():
    text = (
        "#0 0x401123 in Engine::run (this=0x0) at src/engine.cc:15\n"
        "#1 0x401200 in main () at src/main.cc:5\n"
    )

    assert extract_frame_symbols(text) == ["Engine::run", "main"]


def test_addr2line_returns_structured_locations(tmp_path, monkeypatch):
    binary = _binary(tmp_path)

    def fake_run(command, **kwargs):
        if "--version" in command:
            return _fake_version(command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "0x401123\n"
                "Engine::run(int*)\n"
                "/repo/src/engine.cc:15\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("src.analysis.symbol_tools.shutil.which", lambda _name: "/usr/bin/addr2line")
    monkeypatch.setattr("src.analysis.symbol_tools.subprocess.run", fake_run)

    evidence = collect_addr2line_evidence(binary, ["0x401123"])

    assert evidence.status == "ok"
    assert evidence.version == "GNU tool 2.42"
    assert evidence.records == [
        {
            "address": "0x401123",
            "function": "Engine::run(int*)",
            "file_path": "/repo/src/engine.cc",
            "line_number": 15,
            "display": "0x401123 -> Engine::run(int*) at /repo/src/engine.cc:15",
        }
    ]


def test_nm_filters_to_requested_symbol_definitions(tmp_path, monkeypatch):
    binary = _binary(tmp_path)

    def fake_run(command, **kwargs):
        if "--version" in command:
            return _fake_version(command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "0000000000401120 T Engine::run(int*)\t/repo/src/engine.cc:14\n"
                "0000000000401180 T unrelated()\t/repo/src/other.cc:2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("src.analysis.symbol_tools.shutil.which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr("src.analysis.symbol_tools.subprocess.run", fake_run)

    evidence = collect_nm_evidence(binary, ["Engine::run"])

    assert evidence.status == "ok"
    assert len(evidence.records) == 1
    assert evidence.records[0]["query"] == "Engine::run"
    assert "engine.cc:14" in evidence.records[0]["definition"]


def test_cxxfilt_records_demangled_symbols(monkeypatch):
    def fake_run(command, **kwargs):
        if "--version" in command:
            return _fake_version(command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Engine::run(int*)\n",
            stderr="",
        )

    monkeypatch.setattr("src.analysis.symbol_tools.shutil.which", lambda _name: "/usr/bin/c++filt")
    monkeypatch.setattr("src.analysis.symbol_tools.subprocess.run", fake_run)

    evidence = collect_cxxfilt_evidence(["_ZN6Engine3runEPi"])

    assert evidence.status == "ok"
    assert evidence.records[0]["demangled"] == "Engine::run(int*)"


def test_addr2line_reports_no_input_without_main_tool_run(tmp_path, monkeypatch):
    binary = _binary(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _fake_version(command, **kwargs)

    monkeypatch.setattr("src.analysis.symbol_tools.shutil.which", lambda _name: "/usr/bin/addr2line")
    monkeypatch.setattr("src.analysis.symbol_tools.subprocess.run", fake_run)

    evidence = collect_addr2line_evidence(binary, ["not-an-address"])

    assert evidence.status == "no_input"
    assert calls == [["/usr/bin/addr2line", "--version"]]


def test_symbol_tool_reports_unavailable(monkeypatch):
    monkeypatch.setattr("src.analysis.symbol_tools.shutil.which", lambda _name: None)

    evidence = collect_cxxfilt_evidence(["_ZN6Engine3runEPi"])

    assert evidence.status == "unavailable"
    assert evidence.available is False

def test_native_symbol_orchestration_runs_only_relevant_collectors(monkeypatch, tmp_path):
    binary = _binary(tmp_path)
    observed = []

    def fake_addr(_binary, addresses):
        observed.append(("addr2line", list(addresses)))
        return SymbolToolEvidence(
            tool="addr2line",
            status="no_input",
            available=True,
            command=[],
        )

    def fake_nm(_binary, symbols):
        observed.append(("nm", list(symbols)))
        return SymbolToolEvidence(
            tool="nm",
            status="no_input",
            available=True,
            command=[],
        )

    monkeypatch.setattr("src.analysis.symbol_tools.collect_addr2line_evidence", fake_addr)
    monkeypatch.setattr("src.analysis.symbol_tools.collect_nm_evidence", fake_nm)

    evidence = collect_native_symbol_evidence(
        binary,
        "#0 0x401123 in Engine::run () at src/engine.cc:15\n",
    )

    assert observed == [
        ("addr2line", ["0x401123"]),
        ("nm", ["Engine::run"]),
    ]
    assert [item.tool for item in evidence] == ["addr2line", "nm"]


def test_symbol_records_text_deduplicates_display_lines():
    evidence = SymbolToolEvidence(
        tool="c++filt",
        status="ok",
        available=True,
        command=["c++filt"],
        records=[
            {"display": "_Z3foov -> foo()"},
            {"display": "_Z3foov -> foo()"},
        ],
    )

    assert symbol_records_text([evidence]) == "_Z3foov -> foo()"
