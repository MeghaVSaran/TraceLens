"""Run the same retrieval evaluation independently across multiple C++ repositories.

This runner assumes each repository has already been indexed with DebugAid. It does
not clone or build repositories, so it is safe to use after a Colab indexing phase.

Example:
    python scripts/evaluate_repo_matrix.py \
        --manifest data/real_eval/manifest.json \
        --repos-root /tmp \
        --output-dir data/multi_repo_eval \
        --strict-pathless \
        --skip-missing-ground-truth
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "real_eval" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "multi_repo_eval"
MAX_OUTPUT_CHARS = 4000


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a repository evaluation manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("repos"), list):
        raise ValueError("Manifest must be an object containing a 'repos' list.")
    return payload


def _resolve_dataset(manifest_path: Path, value: str) -> Path:
    """Resolve a manifest dataset path relative to the project when possible."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate
    return manifest_path.parent / candidate


def _tail(value: str) -> str:
    """Keep matrix reports bounded even when a build emits a large traceback."""
    value = value or ""
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    return "...\n" + value[-MAX_OUTPUT_CHARS:]


def run_repo_ablation(
    *,
    dataset: Path,
    repo: Path,
    output_dir: Path,
    strict_pathless: bool,
    skip_missing_ground_truth: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one ablation subprocess and load its structured report."""
    command = [
        sys.executable,
        "-m",
        "src.evaluation.run_ablation",
        "--dataset",
        str(dataset),
        "--repo",
        str(repo),
        "--output-dir",
        str(output_dir),
    ]
    if strict_pathless:
        command.append("--strict-pathless")
    if skip_missing_ground_truth:
        command.append("--skip-missing-ground-truth")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "return_code": None,
            "command": command,
            "stdout": _tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr": _tail(exc.stderr if isinstance(exc.stderr, str) else ""),
        }

    record: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": command,
        "stdout": _tail(completed.stdout),
        "stderr": _tail(completed.stderr),
    }
    report_path = output_dir / "ablation_results.json"
    strict_path = output_dir / "ablation_results_strict_no_path.json"
    if completed.returncode == 0 and report_path.exists():
        record["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        record["overall"] = record["report"].get("overall", {})
        if strict_path.exists():
            strict_report = json.loads(strict_path.read_text(encoding="utf-8"))
            record["strict_no_path"] = strict_report.get("overall", {})
    elif completed.returncode == 0:
        record["status"] = "failed"
        record["error"] = "Ablation exited successfully but did not write ablation_results.json."
    return record


def evaluate_manifest(
    *,
    manifest_path: Path,
    repos_root: Path,
    output_dir: Path,
    strict_pathless: bool = False,
    skip_missing_ground_truth: bool = False,
    timeout_seconds: float = 1800.0,
    selected_slugs: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Evaluate every selected manifest repository and preserve honest statuses."""
    manifest_path = Path(manifest_path).resolve()
    repos_root = Path(repos_root).resolve()
    output_dir = Path(output_dir).resolve()
    selected = set(selected_slugs or [])
    repositories: list[dict[str, Any]] = []
    manifest = load_manifest(manifest_path)

    for entry in manifest["repos"]:
        slug = str(entry.get("slug") or "").strip()
        if selected and slug not in selected:
            continue
        repo_path = repos_root / slug
        dataset_path = _resolve_dataset(manifest_path, str(entry.get("dataset", "")))
        result: dict[str, Any] = {
            "repo": entry.get("repo", slug),
            "slug": slug,
            "profile": entry.get("profile", ""),
            "expected_samples": entry.get("samples", 0),
            "dataset": str(dataset_path),
            "repo_path": str(repo_path),
        }
        if not slug:
            result.update({"status": "invalid_manifest", "error": "Missing repository slug."})
        elif not repo_path.is_dir():
            result.update({"status": "missing_repo", "error": "Repository directory not found."})
        elif not dataset_path.is_file():
            result.update({"status": "missing_dataset", "error": "Dataset file not found."})
        elif not (repo_path / ".debugaid").is_dir():
            result.update({"status": "missing_index", "error": "Repository has no .debugaid index."})
        else:
            repo_output = output_dir / slug
            result.update(run_repo_ablation(
                dataset=dataset_path,
                repo=repo_path,
                output_dir=repo_output,
                strict_pathless=strict_pathless,
                skip_missing_ground_truth=skip_missing_ground_truth,
                timeout_seconds=timeout_seconds,
            ))
            result["output_dir"] = str(repo_output)
        repositories.append(result)

    status_counts: dict[str, int] = {}
    for result in repositories:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "manifest": str(manifest_path),
        "repos_root": str(repos_root),
        "output_dir": str(output_dir),
        "strict_pathless": strict_pathless,
        "skip_missing_ground_truth": skip_missing_ground_truth,
        "summary": {
            "total_repositories": len(repositories),
            "successful_repositories": status_counts.get("ok", 0),
            "failed_repositories": len(repositories) - status_counts.get("ok", 0),
            "status_counts": status_counts,
        },
        "repositories": repositories,
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Render a compact matrix report with explicit unavailable statuses."""
    lines = [
        "# Multi-Repository Evaluation",
        "",
        f"- Strict pathless: **{report['strict_pathless']}**",
        f"- Successful repositories: **{report['summary']['successful_repositories']}**",
        f"- Total repositories: **{report['summary']['total_repositories']}**",
        "",
        "| Repository | Profile | Status | N | R@1 | R@5 | MRR | Strict R@5 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["repositories"]:
        overall = item.get("overall", {}).get("hybrid_no_path_boost", {})
        strict = item.get("strict_no_path", {}).get("hybrid_no_path_boost", {})
        values = [
            item.get("repo", item.get("slug", "")),
            item.get("profile", ""),
            item.get("status", "unknown"),
            str(overall.get("n", "-")),
            _metric(overall, "recall_at_1"),
            _metric(overall, "recall_at_5"),
            _metric(overall, "mrr"),
            _metric(strict, "recall_at_5"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repos-root", type=Path, required=True, help="Directory containing indexed repo slugs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-slug", action="append", default=None, help="Evaluate only this manifest slug; repeatable.")
    parser.add_argument("--strict-pathless", action="store_true")
    parser.add_argument("--skip-missing-ground-truth", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    report = evaluate_manifest(
        manifest_path=args.manifest,
        repos_root=args.repos_root,
        output_dir=output_dir,
        strict_pathless=args.strict_pathless,
        skip_missing_ground_truth=args.skip_missing_ground_truth,
        timeout_seconds=args.timeout_seconds,
        selected_slugs=args.repo_slug,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "matrix_results.json"
    markdown_path = output_dir / "matrix_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()