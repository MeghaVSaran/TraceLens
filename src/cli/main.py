"""
DebugAid CLI — main entry point.

Commands:
  debugaid index  --repo PATH [--force-reindex]
  debugaid query  --log PATH --repo PATH [--top-k N] [--verbose] [--output json|text]
  debugaid watch  --repo PATH -- <build-or-test command>
  debugaid eval   --dataset PATH --repo PATH
  debugaid info   --repo PATH

See docs/5_user_stories.md for expected behavior.
"""

import click
import json
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEBUGAID_DIR = ".debugaid"
CHROMA_DIR = "chroma"
BM25_FILE = "bm25.pkl"
METADATA_FILE = "index_meta.json"


def _require_index(repo_path: Path) -> Path:
    """Return the debugaid index path or exit with a helpful message."""
    debugaid_path = repo_path / DEBUGAID_DIR
    if not debugaid_path.exists():
        click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
        sys.exit(1)
    return debugaid_path


def _load_index_metadata(debugaid_path: Path) -> dict:
    """Load index metadata or return empty dict."""
    meta_path = debugaid_path / METADATA_FILE
    if meta_path.exists():
        return json.load(open(meta_path, "r", encoding="utf-8"))
    return {}


def _save_index_metadata(debugaid_path: Path, meta: dict) -> None:
    """Save index metadata."""
    meta_path = debugaid_path / METADATA_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_compile_commands_index(repo_path: Path, build_dir: str | None = None):
    """Best-effort load of compile_commands.json for build-aware diagnosis."""
    from src.analysis.compile_commands import CompileCommandsIndex, discover_compile_commands

    compile_commands_path = discover_compile_commands(
        repo_path,
        build_dir=Path(build_dir).resolve() if build_dir else None,
    )
    if compile_commands_path is None:
        return None, None
    return (
        CompileCommandsIndex.from_file(compile_commands_path, repo_root=repo_path),
        compile_commands_path,
    )


def _triage_log(
    repo_path: Path,
    log_text: str,
    top_k: int,
    build_dir: str | None = None,
    diagnose: bool = False,
):
    """Shared log-to-results pipeline used by query and watch modes."""
    debugaid_path = _require_index(repo_path)

    from src.ingestion.log_parser import parse_log, extract_source_paths
    from src.indexing.vector_index import VectorIndex
    from src.indexing.bm25_index import BM25Index
    from src.embeddings.log_embedder import LogEmbedder
    from src.retrieval.hybrid_retriever import HybridRetriever

    parsed_log = parse_log(log_text, repo_root=repo_path)
    source_paths = parsed_log.source_paths or extract_source_paths(log_text, repo_root=repo_path)

    vector_index = VectorIndex(debugaid_path / CHROMA_DIR)
    bm25_index = BM25Index()
    bm25_index.load(debugaid_path / BM25_FILE)

    log_embedder = LogEmbedder()
    log_embedding = log_embedder.embed_log(parsed_log)

    retriever = HybridRetriever(vector_index, bm25_index)
    results = retriever.retrieve(
        log_embedding,
        parsed_log.query_text(),
        top_k=top_k,
        source_paths=source_paths,
        parsed_log=parsed_log,
    )

    diagnosis = None
    compile_commands_path = None
    if diagnose:
        from src.analysis.cmake_index import CMakeProjectIndex
        from src.analysis.diagnoser import FailureDiagnoser

        compile_commands, compile_commands_path = _load_compile_commands_index(
            repo_path,
            build_dir=build_dir,
        )
        cmake_index = CMakeProjectIndex.from_repo(repo_path)
        diagnoser = FailureDiagnoser(
            repo_root=repo_path,
            compile_commands=compile_commands,
            cmake_index=cmake_index,
        )
        diagnosis = diagnoser.diagnose(parsed_log, results)

    return parsed_log, results, diagnosis, compile_commands_path, source_paths


def _format_diagnosis_text(diagnosis) -> list[str]:
    """Render a diagnosis block for terminal output."""
    if diagnosis is None:
        return []
    lines = [
        "Diagnosis:",
        f"  Summary:         {diagnosis.summary}",
        f"  Likely cause:    {diagnosis.likely_cause}",
        f"  Suggested fix:   {diagnosis.suggested_fix}",
        f"  Confidence:      {diagnosis.confidence}",
    ]
    if diagnosis.compile_command_file:
        lines.append(f"  Compile context: {diagnosis.compile_command_file}")
    if diagnosis.evidence:
        lines.append("  Evidence:")
        for item in diagnosis.evidence:
            lines.append(f"    - {item}")
    return lines


def _format_retrieval_signals_text(parsed_log, source_paths=None) -> list[str]:
    """Render parsed retrieval signals for demo/debug output."""
    source_paths = list(source_paths or getattr(parsed_log, "source_paths", []) or [])
    fields = [
        ("Error type", getattr(parsed_log, "error_type", "")),
        ("Identifiers", ", ".join(getattr(parsed_log, "identifiers", [])[:5])),
        ("Source paths", ", ".join(source_paths[:5])),
        ("File hints", ", ".join(getattr(parsed_log, "file_hints", [])[:5])),
        ("Build targets", ", ".join(getattr(parsed_log, "build_targets", [])[:5])),
    ]
    stack_frames = list(getattr(parsed_log, "stack_frames", []) or [])
    if stack_frames:
        fields.append(("Stack frames", f"{len(stack_frames)} frame(s), first: {stack_frames[0][:100]}"))

    lines = ["Parsed retrieval signals:"]
    for label, value in fields:
        lines.append(f"  {label:<12}: {value or '-'}")
    lines.append("  Retrieval   : BM25 lexical + dense vector + symbol/path-aware hybrid ranking")
    return lines


def _run_command_capture(command: tuple[str, ...], stream_output: bool = True) -> tuple[int, str]:
    """Run a subprocess, optionally streaming merged stdout/stderr."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        if stream_output:
            click.echo(line.rstrip("\n"))
    return_code = process.wait()
    return return_code, "".join(output_lines)


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
def cli():
    """DebugAid — map C++ error logs to relevant source code."""
    pass


# ------------------------------------------------------------------
# INDEX command
# ------------------------------------------------------------------

@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
@click.option("--force-reindex", is_flag=True, default=False, help="Rebuild index even if it exists.")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device for embedding.")
@click.option("--include-tests", is_flag=True, default=False, help="Include test/benchmark files in index.")
@click.option("--embedding-model", default="mpnet", type=click.Choice(["mpnet", "graphcodebert"]),
              help="Embedding backend (mpnet=shared space, graphcodebert=original).")
def index(repo, force_reindex, device, include_tests, embedding_model):
    """Index a C++ repository for log-to-code retrieval."""
    try:
        import torch

        # Auto-detect / validate device.
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                click.echo(f"Using device: cuda ({gpu_name})")
            else:
                click.echo("Warning: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        if device == "cpu":
            click.echo("Using device: cpu")

        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        # Check for existing index.
        if debugaid_path.exists() and not force_reindex:
            click.echo("Index exists. Use --force-reindex to rebuild.")
            sys.exit(0)

        click.echo(f"Indexing repository: {repo_path}")
        t_start = time.time()

        # 1. Parse repository.
        from src.ingestion.code_parser import parse_repository

        click.echo("Parsing C++ files...")
        if not include_tests:
            click.echo("  (excluding test/benchmark files; use --include-tests to keep them)")
        chunks = parse_repository(repo_path, include_tests=include_tests)
        if not chunks:
            click.echo("No C++ files found in repository.", err=True)
            sys.exit(1)
        click.echo(f"  Found {len(chunks)} functions in {repo_path}")

        # 2. Embed chunks.
        from src.embeddings.code_embedder import CodeEmbedder

        click.echo(f"Embedding code chunks with {embedding_model} backend...")
        embedder = CodeEmbedder(backend=embedding_model, device=device)
        embeddings = embedder.embed_chunks(chunks)
        click.echo(f"  Embedded {len(embeddings)} chunks")

        # 3. Build vector index.
        from src.indexing.vector_index import VectorIndex

        debugaid_path.mkdir(parents=True, exist_ok=True)
        chroma_path = debugaid_path / CHROMA_DIR

        click.echo("Building vector index...")
        vector_index = VectorIndex(chroma_path)
        vector_index.build(chunks, embeddings)

        # 4. Build BM25 index.
        from src.indexing.bm25_index import BM25Index

        click.echo("Building BM25 index...")
        bm25_index = BM25Index()
        bm25_index.build(chunks)
        bm25_index.save(debugaid_path / BM25_FILE)

        # 5. Save index metadata.
        _save_index_metadata(debugaid_path, {
            "embedding_backend": embedder.embedding_backend,
            "model_name": embedder.model_name,
            "num_chunks": len(chunks),
            "include_tests": include_tests,
            "indexed_at": datetime.now().isoformat(),
        })

        elapsed = time.time() - t_start
        click.echo(f"Indexed {len(chunks)} chunks in {elapsed:.1f} seconds")
        click.echo(f"  Embedding backend: {embedding_model}")

    except Exception as exc:
        click.echo(f"Error during indexing: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# QUERY command
# ------------------------------------------------------------------

@cli.command()
@click.option("--log", "log_path", required=True, type=click.Path(exists=True), help="Path to log file.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--top-k", default=5, show_default=True, help="Number of results to return.")
@click.option("--verbose", is_flag=True, default=False, help="Show scores and matched code snippets.")
@click.option("--explain-retrieval", is_flag=True, default=False, help="Show parsed log signals and score components for demos/debugging.")
@click.option("--output", default="text", type=click.Choice(["text", "json"]), help="Output format.")
@click.option("--diagnose", is_flag=True, default=False, help="Add build-aware diagnosis if compile_commands.json is available.")
@click.option("--build-dir", default=None, type=click.Path(exists=True), help="Optional build directory containing compile_commands.json.")
def query(log_path, repo, top_k, verbose, explain_retrieval, output, diagnose, build_dir):
    """Query: given a log file, return the most likely source files."""
    try:
        repo_path = Path(repo).resolve()
        _require_index(repo_path)

        # 1. Read log file.
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")

        parsed_log, results, diagnosis, compile_commands_path, source_paths = _triage_log(
            repo_path=repo_path,
            log_text=log_text,
            top_k=top_k,
            build_dir=build_dir,
            diagnose=diagnose,
        )

        if not results:
            click.echo("No results found.", err=True)
            sys.exit(1)

        # 6. Output.
        if output == "json":
            json_results = [
                {
                    "rank": r.rank,
                    "file_path": r.file_path,
                    "function_name": r.function_name,
                    "start_line": r.start_line,
                    "score": round(r.score, 4),
                    "dense_score": round(r.dense_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                    "symbol_score": round(r.symbol_score, 4),
                }
                for r in results
            ]
            if diagnose:
                payload = {
                    "error_type": parsed_log.error_type,
                    "query": parsed_log.query_text(),
                    "compile_commands_path": str(compile_commands_path) if compile_commands_path else "",
                    "diagnosis": diagnosis.to_dict() if diagnosis else None,
                    "results": json_results,
                }
                click.echo(json.dumps(payload, indent=2))
            else:
                click.echo(json.dumps(json_results, indent=2))
        else:
            click.echo(f"\nError type: {parsed_log.error_type}")
            click.echo(f"Query: {parsed_log.query_text()[:120]}")
            if diagnose and compile_commands_path:
                click.echo(f"Compile DB: {compile_commands_path}")
            if explain_retrieval:
                click.echo()
                for line in _format_retrieval_signals_text(parsed_log, source_paths=source_paths):
                    click.echo(line)
            click.echo(f"\nTop {len(results)} results:\n")

            for r in results:
                click.echo(f"  #{r.rank}  {r.file_path}:{r.start_line}")
                click.echo(f"       Function: {r.function_name}")
                if verbose or explain_retrieval:
                    click.echo(f"       Score: {r.score:.4f} "
                               f"(dense={r.dense_score:.4f}, bm25={r.bm25_score:.4f}, "
                               f"symbol={r.symbol_score:.4f})")
                click.echo()
            if diagnose:
                click.echo()
                for line in _format_diagnosis_text(diagnosis):
                    click.echo(line)

    except Exception as exc:
        click.echo(f"Error during query: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# WATCH command
# ------------------------------------------------------------------

@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--top-k", default=5, show_default=True, help="Number of ranked files to surface.")
@click.option("--build-dir", default=None, type=click.Path(exists=True), help="Optional build directory containing compile_commands.json.")
@click.option("--output", default="text", type=click.Choice(["text", "json"]), help="Output format.")
@click.option("--quiet-build-output", is_flag=True, default=False, help="Do not stream command output live before diagnosis.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def watch(repo, top_k, build_dir, output, quiet_build_output, command):
    """Run a build/test command, then triage failures automatically."""
    try:
        repo_path = Path(repo).resolve()
        _require_index(repo_path)

        if not command:
            click.echo("No command provided. Usage: debugaid watch --repo PATH -- <build command>", err=True)
            sys.exit(1)

        click.echo(f"Watching command: {' '.join(command)}")
        return_code, log_text = _run_command_capture(
            tuple(command),
            stream_output=not quiet_build_output and output == "text",
        )

        if return_code == 0:
            if output == "json":
                click.echo(json.dumps({
                    "status": "success",
                    "return_code": return_code,
                    "command": list(command),
                }, indent=2))
            else:
                click.echo("\nCommand completed successfully. No failure triage needed.")
            return

        parsed_log, results, diagnosis, compile_commands_path, _source_paths = _triage_log(
            repo_path=repo_path,
            log_text=log_text,
            top_k=top_k,
            build_dir=build_dir,
            diagnose=True,
        )

        if output == "json":
            payload = {
                "status": "failure",
                "return_code": return_code,
                "command": list(command),
                "error_type": parsed_log.error_type,
                "query": parsed_log.query_text(),
                "compile_commands_path": str(compile_commands_path) if compile_commands_path else "",
                "diagnosis": diagnosis.to_dict() if diagnosis else None,
                "results": [
                    {
                        "rank": r.rank,
                        "file_path": r.file_path,
                        "function_name": r.function_name,
                        "start_line": r.start_line,
                        "score": round(r.score, 4),
                        "dense_score": round(r.dense_score, 4),
                        "bm25_score": round(r.bm25_score, 4),
                        "symbol_score": round(r.symbol_score, 4),
                    }
                    for r in results
                ],
            }
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo(f"\nBuild failed with exit code {return_code}")
        click.echo(f"Detected error type: {parsed_log.error_type}")
        if compile_commands_path:
            click.echo(f"Compile DB: {compile_commands_path}")
        click.echo("\nTop results:\n")
        for r in results:
            click.echo(f"  #{r.rank}  {r.file_path}:{r.start_line}")
            click.echo(f"       Function: {r.function_name}")
            click.echo(f"       Score: {r.score:.4f}")
            click.echo()
        for line in _format_diagnosis_text(diagnosis):
            click.echo(line)

    except Exception as exc:
        click.echo(f"Error during watch: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# EVAL command
# ------------------------------------------------------------------

@cli.command("eval")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to ground truth JSON.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--repo-filter", default=None, help="Only eval samples whose relevant_files all start with this prefix (e.g. 'absl/').")
def eval_cmd(dataset, repo, repo_filter):
    """Evaluate retrieval quality on a ground truth dataset."""
    try:
        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        if not debugaid_path.exists():
            click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
            sys.exit(1)

        # 1. Load dataset.
        with open(dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
        click.echo(f"Loaded {len(data)} samples from {dataset}")

        # Apply optional repo filter to drop samples from unindexed repos.
        if repo_filter:
            original_count = len(data)
            data = [
                s for s in data
                if all(f.startswith(repo_filter) for f in s.get("relevant_files", []))
            ]
            click.echo(
                f"Filtered to {len(data)}/{original_count} samples "
                f"matching prefix '{repo_filter}'"
            )
            if not data:
                click.echo("No samples remain after filtering.", err=True)
                sys.exit(1)

        # 2. Load indices.
        from src.indexing.vector_index import VectorIndex
        from src.indexing.bm25_index import BM25Index

        vector_index = VectorIndex(debugaid_path / CHROMA_DIR)
        bm25_index = BM25Index()
        bm25_index.load(debugaid_path / BM25_FILE)

        # 3. Set up pipeline.
        from src.ingestion.log_parser import parse_log
        from src.embeddings.log_embedder import LogEmbedder
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.evaluation.metrics import evaluate_dataset

        log_embedder = LogEmbedder()
        retriever = HybridRetriever(vector_index, bm25_index)

        # Create a minimal log_parser wrapper that has .parse_log().
        class _LogParserWrapper:
            @staticmethod
            def parse_log(text):
                return parse_log(text)

        click.echo("Running evaluation...")
        t_start = time.time()
        report = evaluate_dataset(data, retriever, _LogParserWrapper(), log_embedder, repo_root=repo_path)
        elapsed = time.time() - t_start

        # 4. Print report.
        filter_label = f" [filtered: {repo_filter}]" if repo_filter else ""
        click.echo(f"\n{'='*60}")
        click.echo(f"  Evaluation Report  ({report.num_samples} samples){filter_label}")
        click.echo(f"{'='*60}")
        if repo_filter:
            click.echo(f"  NOTE: Results are for the {repo_filter}-filtered subset only.")
            click.echo(f"        Do NOT report as overall system performance.")
        click.echo(f"  Recall@1:  {report.recall_at_1:.4f}")
        click.echo(f"  Recall@3:  {report.recall_at_3:.4f}")
        click.echo(f"  Recall@5:  {report.recall_at_5:.4f}")
        click.echo(f"  MRR:       {report.mrr:.4f}")
        click.echo(f"{'='*60}")

        if report.per_error_type:
            click.echo(f"\n  Per Error Type:")
            click.echo(f"  {'Type':<20} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
            click.echo(f"  {'-'*44}")
            for etype, metrics in sorted(report.per_error_type.items()):
                click.echo(
                    f"  {etype:<20} "
                    f"{metrics.get('recall_at_1', 0):.4f} "
                    f"{metrics.get('recall_at_3', 0):.4f} "
                    f"{metrics.get('recall_at_5', 0):.4f} "
                    f"{metrics.get('mrr', 0):.4f}"
                )

        click.echo(f"\n  Completed in {elapsed:.1f} seconds")

    except Exception as exc:
        click.echo(f"Error during evaluation: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# INFO command
# ------------------------------------------------------------------

@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
def info(repo):
    """Show index statistics for a repository."""
    try:
        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        if not debugaid_path.exists():
            click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
            sys.exit(1)

        click.echo(f"\nRepository:  {repo_path}")

        # Chunk count from ChromaDB.
        chroma_path = debugaid_path / CHROMA_DIR
        chunk_count = "unknown"
        if chroma_path.exists():
            try:
                from src.indexing.vector_index import VectorIndex, COLLECTION_NAME
                vi = VectorIndex(chroma_path)
                col = vi._client.get_collection(COLLECTION_NAME)
                chunk_count = col.count()
            except Exception:
                pass
        click.echo(f"Chunks:      {chunk_count}")

        meta = _load_index_metadata(debugaid_path)
        if meta:
            include_tests = meta.get("include_tests")
            if include_tests is not None:
                click.echo(f"Include tests:{' yes' if include_tests else ' no'}")

        # Index size on disk.
        total_size = sum(
            f.stat().st_size for f in debugaid_path.rglob("*") if f.is_file()
        )
        if total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        click.echo(f"Index size:  {size_str}")

        # Date built (modification time of BM25 file).
        bm25_path = debugaid_path / BM25_FILE
        if bm25_path.exists():
            mtime = bm25_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"Date built:  {date_str}")
        else:
            click.echo("Date built:  unknown")

        click.echo()

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
