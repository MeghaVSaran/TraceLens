"""Linux integration proof for addr2line, nm, and c++filt evidence."""

from pathlib import Path
import platform
import shutil
import subprocess

import pytest

from src.analysis.debugger_tools import collect_gdb_core_evidence
from src.analysis.symbol_tools import (
    collect_cxxfilt_evidence,
    collect_native_symbol_evidence,
)


REQUIRED_TOOLS = ("g++", "gdb", "addr2line", "nm", "c++filt")
pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or any(not shutil.which(tool) for tool in REQUIRED_TOOLS),
    reason="requires Linux, g++, gdb, addr2line, nm, and c++filt",
)


CPP_SOURCE = r"""
class Engine {
public:
    __attribute__((noinline)) int run(int* value) {
        return *value;
    }
};

int main() {
    Engine engine;
    return engine.run(nullptr);
}
"""


def test_native_tools_resolve_real_crash_symbols(tmp_path: Path):
    source = tmp_path / "symbol_fixture.cpp"
    binary = tmp_path / "symbol_fixture"
    core = tmp_path / "core.symbols"
    source.write_text(CPP_SOURCE, encoding="utf-8")

    subprocess.run(
        [
            shutil.which("g++"),
            "-std=c++17",
            "-g",
            "-O0",
            "-fno-omit-frame-pointer",
            "-no-pie",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    generated = subprocess.run(
        [
            shutil.which("gdb"),
            "--batch",
            "--quiet",
            "--nx",
            "-ex",
            "set pagination off",
            "-ex",
            "run",
            "-ex",
            f"generate-core-file {core}",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert core.is_file(), generated.stdout + generated.stderr

    gdb = collect_gdb_core_evidence(binary, core, timeout_seconds=30)
    assert gdb.status == "ok", gdb.stderr

    evidence = collect_native_symbol_evidence(binary, gdb.stdout)
    by_tool = {item.tool: item for item in evidence}

    assert by_tool["addr2line"].status == "ok"
    assert any(
        record["file_path"].endswith("symbol_fixture.cpp")
        and record["function"].startswith("Engine::run")
        for record in by_tool["addr2line"].records
    )
    assert by_tool["nm"].status == "ok"
    assert any(
        record["query"] == "Engine::run"
        and "symbol_fixture.cpp" in record["definition"]
        for record in by_tool["nm"].records
    )

    demangled = collect_cxxfilt_evidence(["_ZN6Engine3runEPi"])
    assert demangled.status == "ok"
    assert demangled.records[0]["demangled"].startswith("Engine::run")
