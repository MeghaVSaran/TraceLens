"""Tests for Phase 2 watch-mode CLI behavior."""

from dataclasses import dataclass

from click.testing import CliRunner

from src.cli.main import cli


@dataclass
class _FakeParsedLog:
    error_type: str = "linker_error"
    identifiers: list[str] = None
    source_paths: list[str] = None
    file_hints: list[str] = None
    build_targets: list[str] = None
    stack_frames: list[str] = None

    def __post_init__(self):
        self.identifiers = self.identifiers or ["absl::StrCat"]
        self.source_paths = self.source_paths or []
        self.file_hints = self.file_hints or []
        self.build_targets = self.build_targets or []
        self.stack_frames = self.stack_frames or []

    def query_text(self):
        return "undefined reference to absl::StrCat"


@dataclass
class _FakeRetrievalResult:
    rank: int = 1
    file_path: str = "absl/strings/str_cat.cc"
    function_name: str = "absl::StrCat"
    start_line: int = 42
    score: float = 0.9
    dense_score: float = 0.3
    bm25_score: float = 0.8
    symbol_score: float = 0.2


def test_watch_requires_command(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / ".debugaid").mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--repo", str(repo_root)])

    assert result.exit_code != 0
    assert "No command provided" in result.output


def test_watch_success_does_not_triage(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / ".debugaid").mkdir(parents=True)

    def fake_run(command, stream_output=True):
        return 0, "build ok\n"

    monkeypatch.setattr("src.cli.main._run_command_capture", fake_run)

    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--repo", str(repo_root), "--", "python", "-V"])

    assert result.exit_code == 0
    assert "completed successfully" in result.output


def test_query_explain_retrieval_prints_parsed_signals(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / ".debugaid").mkdir(parents=True)
    log_path = tmp_path / "log.txt"
    log_path.write_text("/usr/bin/ld: undefined reference to `absl::StrCat'\n", encoding="utf-8")

    def fake_triage(repo_path, log_text, top_k, build_dir=None, diagnose=False):
        return _FakeParsedLog(), [_FakeRetrievalResult()], None, None, []

    monkeypatch.setattr("src.cli.main._triage_log", fake_triage)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "query",
            "--log",
            str(log_path),
            "--repo",
            str(repo_root),
            "--explain-retrieval",
        ],
    )

    assert result.exit_code == 0
    assert "Parsed retrieval signals" in result.output
    assert "Identifiers : absl::StrCat" in result.output
    assert "BM25 lexical + dense vector" in result.output
    assert "dense=0.3000, bm25=0.8000, symbol=0.2000" in result.output
