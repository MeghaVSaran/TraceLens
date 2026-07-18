"""Linux integration proof for real GDB core-dump collection."""

from pathlib import Path
import platform
import shutil
import subprocess

import pytest

from src.analysis.crash_analyzer import analyze_crash
from src.analysis.debugger_tools import collect_gdb_core_evidence
from src.ingestion.log_parser import parse_log


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or not shutil.which("g++") or not shutil.which("gdb"),
    reason="requires Linux, g++, and gdb",
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


def test_real_gdb_core_recovers_crash_frame(tmp_path: Path):
    source = tmp_path / "crash_fixture.cpp"
    binary = tmp_path / "crash_fixture"
    core = tmp_path / "core.debugaid"
    source.write_text(CPP_SOURCE, encoding="utf-8")

    subprocess.run(
        [
            shutil.which("g++"),
            "-std=c++17",
            "-g",
            "-O0",
            "-fno-omit-frame-pointer",
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

    evidence = collect_gdb_core_evidence(binary, core, timeout_seconds=30)
    assert evidence.status == "ok", evidence.stderr
    assert "Engine::run" in evidence.stdout
    assert "crash_fixture.cpp" in evidence.stdout

    parsed = parse_log(evidence.stdout, repo_root=tmp_path)
    report = analyze_crash(parsed, [], repo_root=tmp_path, tool_evidence=[evidence])

    assert report.crash_type == "segfault"
    assert report.crash_frame is not None
    assert report.crash_frame.function == "Engine::run"
    assert report.crash_frame.file_path.endswith("crash_fixture.cpp")
