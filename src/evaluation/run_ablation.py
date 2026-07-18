"""
Ablation Evaluation — compare retrieval configurations rigorously.

Runs 5 retrieval configurations and computes metrics overall, by
error type, and by source-path availability.  Includes win/loss
analysis for path boost and per-sample result logging.

Usage:
    python -m src.evaluation.run_ablation \\
        --dataset data/ground_truth/dev.json \\
        --repo /path/to/abseil

Configurations tested:
    1. bm25_only_no_path_boost
    2. bm25_only_with_path_boost
    3. dense_only_no_path_boost
    4. hybrid_no_path_boost
    5. hybrid_with_path_boost
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {"name": "bm25_only_no_path_boost",   "mode": "bm25",   "path_boost": False},
    {"name": "bm25_only_with_path_boost", "mode": "bm25",   "path_boost": True},
    {"name": "dense_only_no_path_boost",  "mode": "dense",  "path_boost": False},
    {"name": "hybrid_no_path_boost",      "mode": "hybrid", "path_boost": False},
    {"name": "hybrid_with_path_boost",    "mode": "hybrid", "path_boost": True},
]


@dataclass
class SampleResult:
    """Per-sample result for one configuration."""
    sample_id: str
    config_name: str
    error_type: str
    has_source_path: bool
    ground_truth_files: List[str]
    retrieved_files: List[str]
    ground_truth_rank: Optional[int]  # 1-indexed, None if not found
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    path_boost_enabled: bool
    strict_pathless: bool = False
    top_scores: List[Dict] = field(default_factory=list)  # [{file, dense, bm25, fused}]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _has_source_path(log_text: str, repo_root: Optional[Path] = None) -> bool:
    """Check whether the log contains an extractable source file path."""
    from src.ingestion.log_parser import extract_source_paths
    paths = extract_source_paths(log_text, repo_root=repo_root)
    return len(paths) > 0


def _looks_like_path_hint(value: str) -> bool:
    """Return True when a hint string looks like a filesystem path."""
    if not value:
        return False
    candidate = value.strip()
    return (
        "/" in candidate
        or "\\" in candidate
        or bool(re.match(r"^[A-Za-z]:", candidate))
        or candidate.startswith("./")
        or candidate.startswith("../")
    )


def _strip_path_tokens(text: str) -> str:
    """Remove path and source-file hints without preserving their basenames."""
    if not text:
        return text
    kept_tokens: List[str] = []
    for token in text.split():
        core = token.strip("`'\"()[]{}:,;")
        if _looks_like_path_hint(core) or _looks_like_source_filename(core):
            continue
        kept_tokens.append(token)
    return " ".join(kept_tokens).strip()


def _looks_like_source_filename(value: str) -> bool:
    """Return True for a C/C++ filename, including a line/column suffix."""
    candidate = str(value or "").strip().strip("`'\"()[]{}:,;")
    candidate = re.sub(r":\d+(?::\d+)?$", "", candidate)
    return bool(re.search(r"\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)$", candidate, re.IGNORECASE))


def _to_strict_pathless_parsed(parsed):
    """Clone ParsedLog with all path-like hints removed."""
    # Strict mode must not retain the filename extracted from a path.
    cleaned_hints: List[str] = []
    return replace(
        parsed,
        raw_log=_strip_path_tokens(getattr(parsed, "raw_log", "")),
        error_message=_strip_path_tokens(getattr(parsed, "error_message", "")),
        file_hints=cleaned_hints,
        source_paths=[],
    )


def _recall_at_k(predictions: List[str], gt: List[str], k: int) -> float:
    gt_set = set(gt)
    for p in predictions[:k]:
        if p in gt_set:
            return 1.0
    return 0.0


def _mrr(predictions: List[str], gt: List[str]) -> float:
    gt_set = set(gt)
    for rank, p in enumerate(predictions, 1):
        if p in gt_set:
            return 1.0 / rank
    return 0.0


def _ground_truth_rank(predictions: List[str], gt: List[str]) -> Optional[int]:
    gt_set = set(gt)
    for rank, p in enumerate(predictions, 1):
        if p in gt_set:
            return rank
    return None


def _has_all_ground_truth_files(item: Dict, repo_path: Path) -> bool:
    """Return True when every ground-truth file exists under repo_path."""
    gt_files = item.get("relevant_files", [])
    return all((repo_path / rel).exists() for rel in gt_files)


def _is_test_like_path(path: str) -> bool:
    """Return True when a path looks like test or benchmark code."""
    value = str(path or "").replace("\\", "/").lower()
    return any(
        marker in value
        for marker in (
            "_test.", "_unittest.", "test_", "/test/", "/tests/",
            "_benchmark.", "_bench.", "benchmark/",
        )
    )


def _count_test_ground_truth_samples(dataset: List[Dict]) -> int:
    """Count samples whose ground truth includes test/benchmark files."""
    return sum(
        1
        for item in dataset
        if any(_is_test_like_path(path) for path in item.get("relevant_files", []))
    )


def _collect_ground_truth_coverage(
    dataset: List[Dict],
    repo_path: Path,
) -> tuple[int, Counter]:
    """Summarize how many dataset items are fully resolvable in the indexed repo."""
    in_repo = 0
    missing_paths: Counter = Counter()
    for item in dataset:
        gt = item.get("relevant_files", [])
        missing = [rel for rel in gt if not (repo_path / rel).exists()]
        if not missing:
            in_repo += 1
        else:
            for rel in missing:
                missing_paths[rel] += 1
    return in_repo, missing_paths


def run_single_config(
    config: Dict,
    dataset: List[Dict],
    retriever,
    log_parser_mod,
    log_embedder,
    repo_root: Optional[Path] = None,
    strict_pathless: bool = False,
) -> List[SampleResult]:
    """Run a single ablation config on the whole dataset."""
    from src.ingestion.log_parser import extract_source_paths

    mode = config["mode"]
    path_boost = config["path_boost"]
    config_name = config["name"]

    results = []

    for item in dataset:
        log_text = item["log"]
        gt_files = item["relevant_files"]
        error_type = item.get("error_type", "unknown")
        sample_id = item.get("id", "unknown")

        # Parse log
        try:
            parsed = log_parser_mod.parse_log(log_text, repo_root=repo_root)
        except TypeError:
            parsed = log_parser_mod.parse_log(log_text)
        retrieval_parsed = _to_strict_pathless_parsed(parsed) if strict_pathless else parsed
        log_embedding = log_embedder.embed_log(retrieval_parsed)

        # Extract source paths
        source_paths = extract_source_paths(log_text, repo_root=repo_root)
        has_sp = len(source_paths) > 0

        # Retrieve
        retrieval_source_paths = [] if strict_pathless else source_paths
        retrieved = retriever.retrieve(
            log_embedding,
            retrieval_parsed.query_text(),
            top_k=5,
            source_paths=retrieval_source_paths if path_boost else None,
            mode=mode,
            path_boost=path_boost,
            parsed_log=retrieval_parsed,
            strict_pathless=strict_pathless,
        )
        pred_files = [r.file_path for r in retrieved]

        # Score breakdowns for top results
        top_scores = [
            {
                "file": r.file_path,
                "function": r.function_name,
                "dense_score": round(r.dense_score, 4),
                "bm25_score": round(r.bm25_score, 4),
                "symbol_score": round(getattr(r, "symbol_score", 0.0), 4),
                "fused_score": round(r.score, 4),
            }
            for r in retrieved
        ]

        results.append(SampleResult(
            sample_id=sample_id,
            config_name=config_name,
            error_type=error_type,
            has_source_path=has_sp,
            ground_truth_files=gt_files,
            retrieved_files=pred_files,
            ground_truth_rank=_ground_truth_rank(pred_files, gt_files),
            recall_at_1=_recall_at_k(pred_files, gt_files, 1),
            recall_at_3=_recall_at_k(pred_files, gt_files, 3),
            recall_at_5=_recall_at_k(pred_files, gt_files, 5),
            mrr=_mrr(pred_files, gt_files),
            path_boost_enabled=path_boost,
            strict_pathless=strict_pathless,
            top_scores=top_scores,
        ))

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(results: List[SampleResult]) -> Dict:
    """Compute mean metrics over a list of SampleResults."""
    n = len(results)
    if n == 0:
        return {"recall_at_1": 0, "recall_at_3": 0, "recall_at_5": 0, "mrr": 0, "n": 0}
    return {
        "recall_at_1": round(sum(r.recall_at_1 for r in results) / n, 4),
        "recall_at_3": round(sum(r.recall_at_3 for r in results) / n, 4),
        "recall_at_5": round(sum(r.recall_at_5 for r in results) / n, 4),
        "mrr": round(sum(r.mrr for r in results) / n, 4),
        "n": n,
    }


def build_ablation_report(
    all_results: Dict[str, List[SampleResult]],
) -> Dict:
    """Build structured report from all configuration results."""
    report = {
        "overall": {},
        "by_error_type": {},
        "by_source_path": {},
        "win_loss": {},
    }

    # Overall per config
    for config_name, results in all_results.items():
        report["overall"][config_name] = _aggregate_metrics(results)

    # Bucketed by error type
    error_types = set()
    for results in all_results.values():
        for r in results:
            error_types.add(r.error_type)

    for etype in sorted(error_types):
        report["by_error_type"][etype] = {}
        for config_name, results in all_results.items():
            filtered = [r for r in results if r.error_type == etype]
            report["by_error_type"][etype][config_name] = _aggregate_metrics(filtered)

    # Bucketed by source path availability
    for bucket_name, has_sp in [("has_source_path", True), ("no_source_path", False)]:
        report["by_source_path"][bucket_name] = {}
        for config_name, results in all_results.items():
            filtered = [r for r in results if r.has_source_path == has_sp]
            report["by_source_path"][bucket_name][config_name] = _aggregate_metrics(filtered)

    # Dedicated strict pathless bucket: all samples evaluated with path hints stripped.
    report["by_source_path"]["strict_no_path"] = {}
    for config_name, results in all_results.items():
        filtered = [r for r in results if r.strict_pathless]
        report["by_source_path"]["strict_no_path"][config_name] = _aggregate_metrics(filtered)

    # Win/loss: path boost impact (hybrid with vs without)
    if ("hybrid_with_path_boost" in all_results and
            "hybrid_no_path_boost" in all_results):
        with_pb = {r.sample_id: r for r in all_results["hybrid_with_path_boost"]}
        without_pb = {r.sample_id: r for r in all_results["hybrid_no_path_boost"]}
        improved = 0
        worsened = 0
        same = 0
        for sid in with_pb:
            if sid not in without_pb:
                continue
            r5_with = with_pb[sid].recall_at_5
            r5_without = without_pb[sid].recall_at_5
            if r5_with > r5_without:
                improved += 1
            elif r5_with < r5_without:
                worsened += 1
            else:
                same += 1
        report["win_loss"]["path_boost_hybrid"] = {
            "improved": improved,
            "worsened": worsened,
            "same": same,
        }

    # Win/loss: BM25 path boost impact
    if ("bm25_only_with_path_boost" in all_results and
            "bm25_only_no_path_boost" in all_results):
        with_pb = {r.sample_id: r for r in all_results["bm25_only_with_path_boost"]}
        without_pb = {r.sample_id: r for r in all_results["bm25_only_no_path_boost"]}
        improved = 0
        worsened = 0
        same = 0
        for sid in with_pb:
            if sid not in without_pb:
                continue
            r5_with = with_pb[sid].recall_at_5
            r5_without = without_pb[sid].recall_at_5
            if r5_with > r5_without:
                improved += 1
            elif r5_with < r5_without:
                worsened += 1
            else:
                same += 1
        report["win_loss"]["path_boost_bm25"] = {
            "improved": improved,
            "worsened": worsened,
            "same": same,
        }

    return report


def build_strict_no_path_report(
    all_results: Dict[str, List[SampleResult]],
) -> Dict:
    """Build a dedicated report over strict-pathless samples only."""
    report = {
        "overall": {},
        "by_error_type": {},
        "n_total": 0,
    }

    strict_results_by_config: Dict[str, List[SampleResult]] = {
        config_name: [r for r in results if r.strict_pathless]
        for config_name, results in all_results.items()
    }

    for config_name, strict_results in strict_results_by_config.items():
        report["overall"][config_name] = _aggregate_metrics(strict_results)

    error_types = set()
    for results in strict_results_by_config.values():
        for r in results:
            error_types.add(r.error_type)

    for etype in sorted(error_types):
        report["by_error_type"][etype] = {}
        for config_name, strict_results in strict_results_by_config.items():
            filtered = [r for r in strict_results if r.error_type == etype]
            report["by_error_type"][etype][config_name] = _aggregate_metrics(filtered)

    report["n_total"] = next(
        (metrics["n"] for metrics in report["overall"].values()),
        0,
    )
    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_markdown_report(report: Dict, index_meta: Dict) -> str:
    """Generate a markdown report from the structured ablation report."""
    lines = []
    lines.append("# Ablation Evaluation Report\n")

    # Index info
    if index_meta:
        lines.append("## Index Info\n")
        lines.append(f"- **Embedding backend**: {index_meta.get('embedding_backend', 'unknown')}")
        lines.append(f"- **Model**: {index_meta.get('model_name', 'unknown')}")
        lines.append(f"- **Chunks**: {index_meta.get('num_chunks', 'unknown')}")
        lines.append(f"- **Indexed at**: {index_meta.get('indexed_at', 'unknown')}")
        lines.append("")

    # Overall table
    lines.append("## Overall Results\n")
    lines.append(f"| {'Configuration':<35} | {'R@1':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>6} | {'N':>4} |")
    lines.append(f"| {'-'*35} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*4} |")
    for config_name, metrics in report["overall"].items():
        lines.append(
            f"| {config_name:<35} | {metrics['recall_at_1']:>5.3f} | "
            f"{metrics['recall_at_3']:>5.3f} | {metrics['recall_at_5']:>5.3f} | "
            f"{metrics['mrr']:>6.4f} | {metrics['n']:>4} |"
        )
    lines.append("")

    # By source path
    lines.append("## By Source Path Availability\n")
    for bucket, configs in report["by_source_path"].items():
        lines.append(f"### {bucket}\n")
        lines.append(f"| {'Configuration':<35} | {'R@1':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>6} | {'N':>4} |")
        lines.append(f"| {'-'*35} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*4} |")
        for config_name, metrics in configs.items():
            lines.append(
                f"| {config_name:<35} | {metrics['recall_at_1']:>5.3f} | "
                f"{metrics['recall_at_3']:>5.3f} | {metrics['recall_at_5']:>5.3f} | "
                f"{metrics['mrr']:>6.4f} | {metrics['n']:>4} |"
            )
        lines.append("")

    # By error type
    lines.append("## By Error Type\n")
    for etype, configs in report["by_error_type"].items():
        lines.append(f"### {etype}\n")
        lines.append(f"| {'Configuration':<35} | {'R@1':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>6} | {'N':>4} |")
        lines.append(f"| {'-'*35} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*4} |")
        for config_name, metrics in configs.items():
            lines.append(
                f"| {config_name:<35} | {metrics['recall_at_1']:>5.3f} | "
                f"{metrics['recall_at_3']:>5.3f} | {metrics['recall_at_5']:>5.3f} | "
                f"{metrics['mrr']:>6.4f} | {metrics['n']:>4} |"
            )
        lines.append("")

    # Win/loss
    if report.get("win_loss"):
        lines.append("## Path Boost Win/Loss Analysis\n")
        for label, wl in report["win_loss"].items():
            lines.append(f"### {label}\n")
            lines.append(f"- Improved (R@5 increased): **{wl['improved']}** samples")
            lines.append(f"- Worsened (R@5 decreased): **{wl['worsened']}** samples")
            lines.append(f"- Same: **{wl['same']}** samples")
            lines.append("")

    return "\n".join(lines)


def format_markdown_strict_no_path_report(report: Dict, index_meta: Dict) -> str:
    """Generate markdown for strict pathless-only evaluation."""
    lines = []
    lines.append("# Strict Pathless Ablation Report\n")

    if index_meta:
        lines.append("## Index Info\n")
        lines.append(f"- **Embedding backend**: {index_meta.get('embedding_backend', 'unknown')}")
        lines.append(f"- **Model**: {index_meta.get('model_name', 'unknown')}")
        lines.append(f"- **Chunks**: {index_meta.get('num_chunks', 'unknown')}")
        lines.append(f"- **Indexed at**: {index_meta.get('indexed_at', 'unknown')}")
        lines.append("")

    lines.append(f"- **Strict pathless samples**: {report.get('n_total', 0)}")
    lines.append("")
    lines.append("## Overall Strict-No-Path Results\n")
    lines.append(f"| {'Configuration':<35} | {'R@1':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>6} | {'N':>4} |")
    lines.append(f"| {'-'*35} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*4} |")
    for config_name, metrics in report["overall"].items():
        lines.append(
            f"| {config_name:<35} | {metrics['recall_at_1']:>5.3f} | "
            f"{metrics['recall_at_3']:>5.3f} | {metrics['recall_at_5']:>5.3f} | "
            f"{metrics['mrr']:>6.4f} | {metrics['n']:>4} |"
        )
    lines.append("")

    lines.append("## By Error Type (Strict-No-Path)\n")
    for etype, configs in report["by_error_type"].items():
        lines.append(f"### {etype}\n")
        lines.append(f"| {'Configuration':<35} | {'R@1':>5} | {'R@3':>5} | {'R@5':>5} | {'MRR':>6} | {'N':>4} |")
        lines.append(f"| {'-'*35} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*4} |")
        for config_name, metrics in configs.items():
            lines.append(
                f"| {config_name:<35} | {metrics['recall_at_1']:>5.3f} | "
                f"{metrics['recall_at_3']:>5.3f} | {metrics['recall_at_5']:>5.3f} | "
                f"{metrics['mrr']:>6.4f} | {metrics['n']:>4} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ablation evaluation comparing retrieval configurations.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to ground truth JSON (e.g. data/ground_truth/dev.json)",
    )
    parser.add_argument(
        "--repo", required=True,
        help="Path to indexed C++ repository",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: data/)",
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Subset of configs to run (default: all 5)",
    )
    parser.add_argument(
        "--repo-filter", default=None,
        help="Only evaluate samples whose ground-truth files all start with this prefix "
             "(e.g. 'absl/' when indexing abseil).",
    )
    parser.add_argument(
        "--strict-pathless",
        action="store_true",
        help="Strip all path-like hints (source paths + slash-style file hints) "
             "before retrieval and report strict_no_path metrics.",
    )
    parser.add_argument(
        "--skip-missing-ground-truth",
        action="store_true",
        help="Exclude samples whose relevant_files do not exist under --repo "
             "(recommended for fair same-repo evaluation).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    repo_path = Path(args.repo).resolve()
    debugaid_path = repo_path / ".debugaid"

    if not debugaid_path.exists():
        print("No index found. Run 'debugaid index --repo ...' first.", file=sys.stderr)
        sys.exit(1)

    # Load index metadata
    meta_path = debugaid_path / "index_meta.json"
    index_meta = {}
    if meta_path.exists():
        index_meta = json.load(open(meta_path, "r", encoding="utf-8"))
        print(f"Index backend: {index_meta.get('embedding_backend', 'unknown')}")
        print(f"Index model:   {index_meta.get('model_name', 'unknown')}")
    else:
        print("Warning: No index metadata found. Backend compatibility cannot be verified.")

    # Check dense retrieval compatibility
    index_backend = index_meta.get("embedding_backend", "unknown")
    log_embedder_backend = "mpnet"  # LogEmbedder always uses mpnet
    dense_valid = (index_backend == log_embedder_backend)
    include_tests = bool(index_meta.get("include_tests", False))

    if not dense_valid:
        print(f"\n⚠ WARNING: Index was built with '{index_backend}' embeddings, "
              f"but log embedder uses '{log_embedder_backend}'.")
        print("  Dense retrieval results will be INVALID (different embedding spaces).")
        print("  Dense-only and hybrid configs will run but results are unreliable.")
        print("  Re-index with --embedding-model mpnet for valid dense retrieval.\n")

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")

    if args.repo_filter:
        before = len(dataset)
        dataset = [
            item for item in dataset
            if all(path.startswith(args.repo_filter) for path in item.get("relevant_files", []))
        ]
        print(
            f"Filtered dataset by prefix '{args.repo_filter}': "
            f"{len(dataset)}/{before} samples"
        )
        if not dataset:
            print("No samples left after --repo-filter; exiting.", file=sys.stderr)
            sys.exit(1)

    test_gt_samples = _count_test_ground_truth_samples(dataset)
    if test_gt_samples and not include_tests:
        print(
            f"Warning: {test_gt_samples}/{len(dataset)} samples have test/benchmark "
            "ground-truth files, but this index was built without --include-tests."
        )
        print(
            "  Realistic GitHub/test-failure evaluation may be unfairly low; "
            "re-index with --include-tests for that slice."
        )

    in_repo, missing_paths = _collect_ground_truth_coverage(dataset, repo_path)

    if in_repo < len(dataset):
        print(
            f"Warning: only {in_repo}/{len(dataset)} samples have all ground-truth files "
            f"inside indexed repo '{repo_path.name}'."
        )
        top_missing = ", ".join(
            f"{path} ({count}x)"
            for path, count in missing_paths.most_common(5)
        )
        if top_missing:
            print(f"Top missing ground-truth paths: {top_missing}")

    dropped_missing_gt = 0
    if args.skip_missing_ground_truth:
        before = len(dataset)
        dataset = [item for item in dataset if _has_all_ground_truth_files(item, repo_path)]
        dropped_missing_gt = before - len(dataset)
        print(
            f"Skipped {dropped_missing_gt} samples with missing ground-truth files "
            f"(--skip-missing-ground-truth). Remaining: {len(dataset)}"
        )
        if not dataset:
            print("No samples left after skipping missing ground-truth files; exiting.", file=sys.stderr)
            sys.exit(1)

    # Load indices
    from src.indexing.vector_index import VectorIndex
    from src.indexing.bm25_index import BM25Index

    vector_index = VectorIndex(debugaid_path / "chroma")
    bm25_index = BM25Index()
    bm25_index.load(debugaid_path / "bm25.pkl")

    # Set up pipeline
    from src.ingestion import log_parser as log_parser_mod
    from src.embeddings.log_embedder import LogEmbedder
    from src.retrieval.hybrid_retriever import HybridRetriever

    log_embedder = LogEmbedder()
    retriever = HybridRetriever(vector_index, bm25_index)

    # Select configs
    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]
        if not configs:
            print(f"No matching configs. Available: {[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    # Run ablations
    all_results: Dict[str, List[SampleResult]] = {}
    t_start = time.time()

    for config in configs:
        cname = config["name"]
        print(f"\nRunning: {cname} ...")

        # Skip dense configs with incompatible backends (with warning)
        if not dense_valid and config["mode"] in ("dense", "hybrid"):
            print(f"  ⚠ Dense retrieval invalid for this index (backend mismatch)")

        results = run_single_config(
            config, dataset, retriever, log_parser_mod, log_embedder,
            repo_root=repo_path,
            strict_pathless=args.strict_pathless,
        )
        all_results[cname] = results

        # Quick summary
        agg = _aggregate_metrics(results)
        print(f"  R@1={agg['recall_at_1']:.3f}  R@3={agg['recall_at_3']:.3f}  "
              f"R@5={agg['recall_at_5']:.3f}  MRR={agg['mrr']:.4f}")

    elapsed = time.time() - t_start
    print(f"\nAll ablations completed in {elapsed:.1f}s")

    # Build report
    report = build_ablation_report(all_results)
    report["index_metadata"] = index_meta
    report["dense_retrieval_valid"] = dense_valid
    report["dataset_info"] = {
        "loaded_samples": len(dataset) + dropped_missing_gt,
        "evaluated_samples": len(dataset),
        "skipped_missing_ground_truth": dropped_missing_gt,
        "repo_filter": args.repo_filter,
        "strict_pathless": bool(args.strict_pathless),
        "skip_missing_ground_truth": bool(args.skip_missing_ground_truth),
        "index_include_tests": include_tests,
        "test_ground_truth_samples": test_gt_samples,
    }

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report: {json_path}")

    # Save per-sample log
    sample_log_path = output_dir / "ablation_per_sample.json"
    sample_log = {}
    for config_name, results in all_results.items():
        sample_log[config_name] = [asdict(r) for r in results]
    with open(sample_log_path, "w", encoding="utf-8") as f:
        json.dump(sample_log, f, indent=2, ensure_ascii=False)
    print(f"Per-sample log: {sample_log_path}")

    # Save markdown
    md_path = output_dir / "ablation_report.md"
    md_text = format_markdown_report(report, index_meta)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Markdown report: {md_path}")

    if args.strict_pathless:
        strict_report = build_strict_no_path_report(all_results)
        strict_json_path = output_dir / "ablation_results_strict_no_path.json"
        with open(strict_json_path, "w", encoding="utf-8") as f:
            json.dump(strict_report, f, indent=2, ensure_ascii=False)
        print(f"Strict-no-path JSON report: {strict_json_path}")

        strict_md_path = output_dir / "ablation_report_strict_no_path.md"
        strict_md_text = format_markdown_strict_no_path_report(strict_report, index_meta)
        strict_md_path.write_text(strict_md_text, encoding="utf-8")
        print(f"Strict-no-path Markdown report: {strict_md_path}")

    # Print summary table
    print("\n" + md_text)


if __name__ == "__main__":
    main()
