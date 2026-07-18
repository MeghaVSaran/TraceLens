"""Tests for multi-repository evaluation orchestration."""

import json
from types import SimpleNamespace

from scripts.evaluate_repo_matrix import evaluate_manifest, format_markdown


def _write_manifest(tmp_path):
    dataset = tmp_path / "llvm.json"
    dataset.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "repo": "llvm/llvm-project",
                        "slug": "llvm_llvm_project",
                        "profile": "very_large_compiler_infra",
                        "dataset": "llvm.json",
                        "samples": 42,
                    },
                    {
                        "repo": "opencv/opencv",
                        "slug": "opencv_opencv",
                        "profile": "large_runtime_library",
                        "dataset": "llvm.json",
                        "samples": 8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_evaluate_manifest_preserves_success_and_missing_index_status(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    repos_root = tmp_path / "repos"
    indexed_repo = repos_root / "llvm_llvm_project" / ".debugaid"
    indexed_repo.mkdir(parents=True)
    (indexed_repo / "index_meta.json").write_text("{}", encoding="utf-8")
    (indexed_repo / "bm25.pkl").write_bytes(b"index")
    (indexed_repo / "chroma").mkdir()
    (repos_root / "opencv_opencv").mkdir(parents=True)
    output_dir = tmp_path / "reports"

    def fake_run(command, **_kwargs):
        report_dir = tmp_path / "reports" / "llvm_llvm_project"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "ablation_results.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "hybrid_no_path_boost": {
                            "n": 40,
                            "recall_at_1": 0.4,
                            "recall_at_5": 0.8,
                            "mrr": 0.6,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (report_dir / "ablation_results_strict_no_path.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "hybrid_no_path_boost": {
                            "n": 40,
                            "recall_at_5": 0.7,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.evaluate_repo_matrix.subprocess.run", fake_run)
    report = evaluate_manifest(
        manifest_path=manifest,
        repos_root=repos_root,
        output_dir=output_dir,
        strict_pathless=True,
        skip_missing_ground_truth=True,
    )

    assert report["summary"]["successful_repositories"] == 1
    assert report["summary"]["status_counts"]["missing_index"] == 1
    llvm = next(item for item in report["repositories"] if item["slug"] == "llvm_llvm_project")
    assert llvm["overall"]["hybrid_no_path_boost"]["recall_at_5"] == 0.8
    assert llvm["strict_no_path"]["hybrid_no_path_boost"]["recall_at_5"] == 0.7
    assert "very_large_compiler_infra" in format_markdown(report)


def test_repo_selection_limits_matrix(tmp_path):
    manifest = _write_manifest(tmp_path)
    report = evaluate_manifest(
        manifest_path=manifest,
        repos_root=tmp_path / "missing-repos",
        output_dir=tmp_path / "reports",
        selected_slugs=["opencv_opencv"],
    )

    assert [item["slug"] for item in report["repositories"]] == ["opencv_opencv"]
    assert report["repositories"][0]["status"] == "missing_repo"