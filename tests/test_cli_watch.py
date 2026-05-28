"""Tests for Phase 2 watch-mode CLI behavior."""

from click.testing import CliRunner

from src.cli.main import cli


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
