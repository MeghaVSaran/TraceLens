"""Prepare repo-sliced real-world evaluation datasets.

This script audits GitHub-mined DebugAid samples and writes one dataset file per
repository, plus a manifest and markdown report with Colab-ready commands.

Usage:
    python scripts/prepare_real_eval.py
    python scripts/prepare_real_eval.py --input data/processed/github_pairs.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "github_pairs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "real_eval"

CPP_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx")
PATHLIKE_RE = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s:]+?\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)\b|(?:[\w.-]+[\\/])+[\w.-]+\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)\b")


REPO_PROFILES: dict[str, dict[str, str]] = {
    "llvm/llvm-project": {
        "profile": "very_large_compiler_infra",
        "clone_url": "https://github.com/llvm/llvm-project.git",
        "notes": "Closest current proxy for compiler/IR-heavy industrial C++.",
    },
    "opencv/opencv": {
        "profile": "large_runtime_library",
        "clone_url": "https://github.com/opencv/opencv.git",
        "notes": "Large C++ runtime/library codebase with platform and build failures.",
    },
    "abseil/abseil-cpp": {
        "profile": "small_modern_cpp_library",
        "clone_url": "https://github.com/abseil/abseil-cpp.git",
        "notes": "Good fast sanity target, but not enough by itself.",
    },
}


FUTURE_TARGET_REPOS = [
    {
        "repo": "verilator/verilator",
        "profile": "open_source_eda_simulator",
        "why": "EDA-relevant C++ compiler/simulator with realistic build, parser, and runtime failures.",
    },
    {
        "repo": "YosysHQ/yosys",
        "profile": "open_source_synthesis_tool",
        "why": "EDA-relevant synthesis codebase with C++ passes and frontend/backend failures.",
    },
    {
        "repo": "The-OpenROAD-Project/OpenROAD",
        "profile": "open_source_physical_design",
        "why": "Closest public proxy for large EDA toolchains and complex CMake/dependency issues.",
    },
    {
        "repo": "protocolbuffers/protobuf",
        "profile": "large_codegen_runtime",
        "why": "Generated-code and ABI/build failures are common in industrial C++.",
    },
    {
        "repo": "fmtlib/fmt",
        "profile": "small_template_heavy_library",
        "why": "Small but template-heavy, useful for quick pathless compiler-error checks.",
    },
]


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load dataset samples from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [sample for sample in data if isinstance(sample, dict)]


def is_valid_real_sample(sample: dict[str, Any]) -> bool:
    """Return True if a sample is usable for real-world evaluation."""
    if sample.get("source") != "github":
        return False
    if sample.get("error_type") == "unknown":
        return False
    if not sample.get("repo") or not sample.get("log"):
        return False
    relevant_files = sample.get("relevant_files") or []
    if not relevant_files or len(relevant_files) > 10:
        return False
    return all(str(path).endswith(CPP_EXTENSIONS) for path in relevant_files)


def has_pathlike_hint(sample: dict[str, Any]) -> bool:
    """Return True if the raw log appears to contain source path hints."""
    return PATHLIKE_RE.search(str(sample.get("log", ""))) is not None


def repo_slug(repo: str) -> str:
    """Convert owner/name to a filesystem-safe slug."""
    return repo.replace("/", "_").replace("-", "_")


def display_path(path: Path) -> str:
    """Return a stable display path, relative to project root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate counts for a list of samples."""
    repos = Counter(sample.get("repo", "unknown") for sample in samples)
    error_types = Counter(sample.get("error_type", "unknown") for sample in samples)
    pathlike = sum(1 for sample in samples if has_pathlike_hint(sample))
    relevant_file_counts = Counter(len(sample.get("relevant_files") or []) for sample in samples)
    return {
        "n": len(samples),
        "repos": dict(sorted(repos.items())),
        "error_types": dict(sorted(error_types.items())),
        "pathlike_logs": pathlike,
        "strict_pathless_needed": pathlike,
        "relevant_file_counts": dict(sorted(relevant_file_counts.items())),
    }


def group_by_repo(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group samples by repository."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["repo"])].append(sample)
    return dict(sorted(grouped.items()))


def build_manifest(
    grouped: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    input_path: Path,
) -> dict[str, Any]:
    """Create the real-eval manifest structure."""
    repos = []
    for repo, samples in grouped.items():
        slug = repo_slug(repo)
        profile = REPO_PROFILES.get(repo, {})
        repos.append({
            "repo": repo,
            "slug": slug,
            "profile": profile.get("profile", "unclassified_cpp_repo"),
            "clone_url": profile.get("clone_url", f"https://github.com/{repo}.git"),
            "dataset": display_path(output_dir / f"{slug}.json"),
            "samples": len(samples),
            "error_types": dict(sorted(Counter(sample["error_type"] for sample in samples).items())),
            "pathlike_logs": sum(1 for sample in samples if has_pathlike_hint(sample)),
            "notes": profile.get("notes", ""),
        })
    return {
        "source_dataset": display_path(input_path),
        "total_samples": sum(repo["samples"] for repo in repos),
        "repos": repos,
        "future_target_repos": FUTURE_TARGET_REPOS,
    }


def write_outputs(
    grouped: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-repo datasets, manifest, and markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for repo, samples in grouped.items():
        path = output_dir / f"{repo_slug(repo)}.json"
        path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(build_markdown_report(manifest), encoding="utf-8")


def build_markdown_report(manifest: dict[str, Any]) -> str:
    """Render a human-readable real-eval report."""
    lines = [
        "# Real-World Evaluation Manifest",
        "",
        f"- Source dataset: `{manifest['source_dataset']}`",
        f"- Total usable GitHub samples: **{manifest['total_samples']}**",
        "",
        "## Repository Slices",
        "",
        "| Repo | Profile | Samples | Error Types | Dataset |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for repo in manifest["repos"]:
        error_types = ", ".join(f"{key}:{value}" for key, value in repo["error_types"].items())
        lines.append(
            f"| `{repo['repo']}` | {repo['profile']} | {repo['samples']} | {error_types} | `{repo['dataset']}` |"
        )

    lines.extend([
        "",
        "## Colab Evaluation Template",
        "",
        "For each repo, clone it, build a compile database when practical, index it, then run strict-pathless ablation.",
        "",
    ])
    for repo in manifest["repos"]:
        slug = repo["slug"]
        lines.extend([
            f"### {repo['repo']}",
            "",
            "```bash",
            f"git clone --depth=1 {repo['clone_url']} /tmp/{slug}",
            f"python -m src.cli.main index --repo /tmp/{slug} --device cuda --embedding-model mpnet --include-tests --force-reindex",
            f"python -m src.evaluation.run_ablation --dataset {repo['dataset']} --repo /tmp/{slug} --strict-pathless --skip-missing-ground-truth",
            "```",
            "",
        ])

    lines.extend([
        "## Next Mining Targets",
        "",
        "| Repo | Profile | Why |",
        "| --- | --- | --- |",
    ])
    for target in manifest["future_target_repos"]:
        lines.append(f"| `{target['repo']}` | {target['profile']} | {target['why']} |")

    lines.extend([
        "",
        "## Training Gate",
        "",
        "Do not train a model yet. First build a reliable multi-repo eval set with at least 300-500 high-quality real samples, including EDA-like repos. Then compare BM25, dense, hybrid, reranker, and any trained model on held-out repos.",
        "",
    ])
    return "\n".join(lines)


def prepare_real_eval(input_path: Path, output_dir: Path, min_samples: int = 1) -> dict[str, Any]:
    """Prepare datasets and return the manifest."""
    raw_samples = load_samples(input_path)
    usable = [sample for sample in raw_samples if is_valid_real_sample(sample)]
    grouped = {
        repo: samples
        for repo, samples in group_by_repo(usable).items()
        if len(samples) >= min_samples
    }
    manifest = build_manifest(grouped, output_dir=output_dir, input_path=input_path)
    manifest["audit"] = {
        "raw": summarize_samples(raw_samples),
        "usable": summarize_samples(usable),
        "dropped_samples": len(raw_samples) - len(usable),
    }
    write_outputs(grouped, manifest, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-samples", type=int, default=1)
    args = parser.parse_args()

    manifest = prepare_real_eval(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        min_samples=args.min_samples,
    )
    print(f"Prepared {manifest['total_samples']} usable real samples")
    print(f"Repos: {len(manifest['repos'])}")
    for repo in manifest["repos"]:
        print(f"  {repo['repo']}: {repo['samples']} samples -> {repo['dataset']}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    print(f"Report:   {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
