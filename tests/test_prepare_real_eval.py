"""Tests for real-world evaluation dataset preparation."""

import json

from scripts.prepare_real_eval import (
    has_pathlike_hint,
    is_valid_real_sample,
    prepare_real_eval,
    repo_slug,
)


def _sample(repo="llvm/llvm-project", error_type="segfault", files=None, log=None):
    return {
        "id": f"{repo}-1",
        "log": log or "Segmentation fault\n#0 llvm::Foo::bar /src/lib/Foo.cpp:10:3",
        "relevant_files": files or ["llvm/lib/Foo.cpp"],
        "error_type": error_type,
        "source": "github",
        "repo": repo,
        "issue_url": f"https://github.com/{repo}/issues/1",
    }


def test_repo_slug_is_filesystem_safe():
    assert repo_slug("llvm/llvm-project") == "llvm_llvm_project"


def test_valid_real_sample_requires_github_known_type_and_cpp_files():
    assert is_valid_real_sample(_sample()) is True
    assert is_valid_real_sample({**_sample(), "source": "synthetic"}) is False
    assert is_valid_real_sample({**_sample(), "error_type": "unknown"}) is False
    assert is_valid_real_sample({**_sample(), "relevant_files": ["README.md"]}) is False


def test_has_pathlike_hint_detects_source_paths():
    assert has_pathlike_hint(_sample(log="/tmp/project/src/foo.cc:10:1: error: bad")) is True
    assert has_pathlike_hint(_sample(log="undefined reference to `Foo::bar`")) is False


def test_prepare_real_eval_writes_manifest_and_repo_slices(tmp_path):
    input_path = tmp_path / "github_pairs.json"
    output_dir = tmp_path / "real_eval"
    rows = [
        _sample(repo="llvm/llvm-project", error_type="segfault", files=["llvm/lib/Foo.cpp"]),
        _sample(repo="opencv/opencv", error_type="compiler_error", files=["modules/core/src/foo.cpp"]),
        _sample(repo="bad/repo", error_type="unknown", files=["bad.cc"]),
    ]
    input_path.write_text(json.dumps(rows), encoding="utf-8")

    manifest = prepare_real_eval(input_path=input_path, output_dir=output_dir)

    assert manifest["total_samples"] == 2
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "llvm_llvm_project.json").exists()
    assert (output_dir / "opencv_opencv.json").exists()
    assert manifest["audit"]["dropped_samples"] == 1
