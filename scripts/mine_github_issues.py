"""
GitHub Issue Miner — Extract (log, relevant_files) pairs from C++ repos.

Searches public C++ repositories for issues containing build/runtime error
logs and linked fixing commits.  Produces labeled pairs for the DebugAid
evaluation dataset.

Usage:
    export GITHUB_TOKEN=ghp_...
    python scripts/mine_github_issues.py

See docs/4_dataset_strategy.md §1 for the full mining strategy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from github import Github, GithubException, Auth
except ModuleNotFoundError:
    Github = None
    GithubException = Exception
    Auth = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Project root — two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "github_issues"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ERROR_KEYWORDS: List[str] = [
    "undefined reference",
    "undefined symbol",
    "no such file or directory",
    "undeclared identifier",
    "segmentation fault",
    "stack dump",
    "AddressSanitizer",
    "heap-buffer-overflow",
    "use-after-free",
    "no matching function",
    "CMake Error",
    "No rule to make target",
    "used but never defined",
]

# Repos to mine (owner/name)
TARGET_REPOS: List[str] = [
    "llvm/llvm-project",
    "opencv/opencv",
    "abseil/abseil-cpp",
    "verilator/verilator",
    "YosysHQ/yosys",
    "The-OpenROAD-Project/OpenROAD",
    "protocolbuffers/protobuf",
    "fmtlib/fmt",
]

MAX_RETRIES = 3
RATE_LIMIT_SLEEP_SECONDS = 60

# ---------------------------------------------------------------------------
# Error-type classification (mirrors src/ingestion/log_parser.py patterns)
# ---------------------------------------------------------------------------

_ERROR_TYPE_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("include_error", re.compile(
        r"fatal error:\s*\S+\s*:\s*No such file or directory", re.IGNORECASE)),
    ("template_error", re.compile(
        r"(?:implicit instantiation of undefined template|undefined template)",
        re.IGNORECASE)),
    ("asan_error", re.compile(
        r"heap-buffer-overflow on address", re.IGNORECASE)),
    ("asan_error", re.compile(
        r"use-after-free on address", re.IGNORECASE)),
    ("asan_error", re.compile(
        r"stack-buffer-overflow", re.IGNORECASE)),
    ("asan_error", re.compile(
        r"AddressSanitizer", re.IGNORECASE)),
    ("build_system_error", re.compile(
        r"Could not find (?:package|module)\s+", re.IGNORECASE)),
    ("build_system_error", re.compile(
        r"No rule to make target\s+", re.IGNORECASE)),
    ("build_system_error", re.compile(
        r"CMake Error", re.IGNORECASE)),
    ("linker_error", re.compile(
        r"undefined reference to\s+", re.IGNORECASE)),
    ("linker_error", re.compile(
        r"undefined symbol", re.IGNORECASE)),
    ("linker_error", re.compile(
        r"multiple definition of\s+", re.IGNORECASE)),
    ("linker_error", re.compile(
        r"used but never defined", re.IGNORECASE)),
    ("compiler_error", re.compile(
        r"use of undeclared identifier\s+", re.IGNORECASE)),
    ("compiler_error", re.compile(
        r"no matching function for call to\s+", re.IGNORECASE)),
    ("runtime_exception", re.compile(
        r"terminate called after throwing an instance of\s+", re.IGNORECASE)),
    ("runtime_exception", re.compile(
        r"std::(?:bad_alloc|out_of_range|runtime_error|logic_error|invalid_argument)",
        re.IGNORECASE)),
    ("segfault", re.compile(r"Segmentation fault", re.IGNORECASE)),
]


def classify_error_type(log_text: str) -> str:
    """Classify a log snippet into one of the eight error categories.

    Uses the same priority-ordered patterns as ``src/ingestion/log_parser.py``
    but implemented standalone so this script has no internal imports.

    Args:
        log_text: Raw log / code-block text.

    Returns:
        One of: linker_error, compiler_error, include_error,
                template_error, segfault, asan_error,
                build_system_error, runtime_exception, unknown.
    """
    for error_type, pattern in _ERROR_TYPE_PATTERNS:
        if pattern.search(log_text):
            return error_type
    return "unknown"


# ---------------------------------------------------------------------------
# Rate-limit–aware GitHub API helper
# ---------------------------------------------------------------------------

def _api_call_with_retry(func, *args, **kwargs) -> Any:
    """Execute a GitHub API call with rate-limit retry logic.

    Catches ``GithubException`` with status 403 or 429 (rate limit),
    sleeps for ``RATE_LIMIT_SLEEP_SECONDS`` and retries up to
    ``MAX_RETRIES`` times.

    Args:
        func: Callable that performs the API request.
        *args: Positional arguments forwarded to *func*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        The return value of *func*.

    Raises:
        GithubException: If all retries are exhausted or the error
            is not rate-limit related.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except GithubException as exc:
            if exc.status in (403, 429):
                logger.warning(
                    "Rate limited (HTTP %d). Sleeping %ds … (attempt %d/%d)",
                    exc.status, RATE_LIMIT_SLEEP_SECONDS, attempt, MAX_RETRIES,
                )
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            else:
                raise
    # Final attempt — let any exception propagate
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Log-snippet extraction
# ---------------------------------------------------------------------------

_RE_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_log_snippet(body: str) -> Optional[str]:
    """Return the first fenced code block that contains an error keyword.

    Args:
        body: The issue body (Markdown).

    Returns:
        The code block content, or ``None`` if no matching block is found.
    """
    if not body:
        return None
    for match in _RE_CODE_BLOCK.finditer(body):
        block = match.group(1).strip()
        for kw in ERROR_KEYWORDS:
            if kw.lower() in block.lower():
                return block
    return None


# ---------------------------------------------------------------------------
# Fixing-commit detection
# ---------------------------------------------------------------------------

_RE_FIXES = re.compile(
    r"(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+#(\d+)",
    re.IGNORECASE,
)


def _find_fixing_commit(
    repo,  # github.Repository.Repository
    issue,  # github.Issue.Issue
) -> Optional[List[str]]:
    """Search issue comments, events, and commits for a fixing commit.

    Looks for patterns like ``fixes #N`` or ``closes #N``, as well as
    commit SHA references.  If found, retrieves the commit and returns
    the list of modified ``.cpp`` / ``.h`` files.

    Args:
        repo: PyGithub Repository object.
        issue: PyGithub Issue object.

    Returns:
        A list of modified C++/header file paths, or ``None`` if no
        fixing commit was found.
    """
    issue_number = issue.number
    cpp_extensions = (".cpp", ".h", ".hpp", ".cc", ".cxx", ".hxx")

    def _cpp_files_from_commit(commit) -> Optional[List[str]]:
        """Extract C++ file paths from a commit object."""
        try:
            file_list = commit.files or []
            cpp = [f.filename for f in file_list
                   if f.filename.endswith(cpp_extensions)]
            return cpp if cpp else None
        except Exception:
            return None

    # --- 1. Scan issue comments for "fixes/closes #N" or commit SHAs --------
    _RE_SHA = re.compile(r"\b([0-9a-f]{40})\b")
    try:
        comments = _api_call_with_retry(issue.get_comments)
        for comment in comments:
            body = comment.body or ""

            # Check for direct commit SHA references
            for sha_match in _RE_SHA.finditer(body):
                sha = sha_match.group(1)
                try:
                    commit = _api_call_with_retry(repo.get_commit, sha)
                    result = _cpp_files_from_commit(commit)
                    if result:
                        return result
                except GithubException:
                    pass
    except GithubException:
        logger.debug("Could not fetch comments for issue #%d", issue_number)

    # --- 2. Use issue events to find cross-referenced/merged PRs ------------
    try:
        events = _api_call_with_retry(issue.get_events)
        for event in events:
            # "closed" events with a commit_id indicate the closing commit
            if event.event == "closed" and event.commit_id:
                try:
                    commit = _api_call_with_retry(
                        repo.get_commit, event.commit_id,
                    )
                    result = _cpp_files_from_commit(commit)
                    if result:
                        return result
                except GithubException:
                    pass

            # "referenced" events also carry a commit_id
            if event.event == "referenced" and event.commit_id:
                try:
                    commit = _api_call_with_retry(
                        repo.get_commit, event.commit_id,
                    )
                    msg = commit.commit.message or ""
                    # Verify the commit actually references this issue
                    for m in _RE_FIXES.finditer(msg):
                        if int(m.group(1)) == issue_number:
                            result = _cpp_files_from_commit(commit)
                            if result:
                                return result
                except GithubException:
                    pass
    except GithubException:
        logger.debug("Could not fetch events for issue #%d", issue_number)

    # --- 3. If the issue is itself a PR, check its files directly -----------
    if issue.pull_request:
        try:
            pr = _api_call_with_retry(repo.get_pull, issue_number)
            if pr.merged:
                files = _api_call_with_retry(pr.get_files)
                cpp_files = [
                    f.filename for f in files
                    if f.filename.endswith(cpp_extensions)
                ]
                if cpp_files:
                    return cpp_files
        except GithubException:
            logger.debug("Could not fetch PR #%d files", issue_number)

    # --- 4. Bounded commit search for "fixes #N" in recent commits ----------
    try:
        commits = _api_call_with_retry(repo.get_commits)
        checked = 0
        for commit in commits:
            msg = commit.commit.message or ""
            for m in _RE_FIXES.finditer(msg):
                ref_num = int(m.group(1))
                if ref_num == issue_number:
                    result = _cpp_files_from_commit(commit)
                    if result:
                        return result
            checked += 1
            if checked >= 200:
                break
    except GithubException:
        logger.debug("Could not search commits for issue #%d", issue_number)

    return None


# ---------------------------------------------------------------------------
# Incremental persistence
# ---------------------------------------------------------------------------

def _load_existing_results(path: Path) -> List[Dict]:
    """Load previously-saved results from a JSON file.

    Returns an empty list if the file does not exist or is invalid.
    """
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                logger.info("Loaded %d existing results from %s", len(data), path)
                return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load %s — starting fresh", path)
    return []


def _save_results(results: List[Dict], path: Path) -> None:
    """Atomically write results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Core mining function
# ---------------------------------------------------------------------------

def mine_repo(
    repo_name: str,
    max_issues: int = 200,
    github_client: Optional[Github] = None,
) -> List[Dict]:
    """Mine a GitHub repository for labeled (log, relevant_files) pairs.

    Searches issues labelled ``bug`` or whose body contains C++ error
    keywords.  For each qualifying issue, extracts a log snippet and
    identifies the fixing commit's modified C++/header files.

    Results are saved **incrementally** to
    ``data/raw/github_issues/{repo_name_safe}.json`` after every
    successful issue to prevent data loss on interruption.

    Args:
        repo_name: GitHub repository in ``owner/name`` format.
        max_issues: Maximum number of issues to process.
        github_client: Optional pre-configured ``Github`` instance.
            Falls back to ``GITHUB_TOKEN`` environment variable.

    Returns:
        List of result dicts matching the schema in
        ``docs/4_dataset_strategy.md``.
    """
    # ---- GitHub client setup -----------------------------------------------
    if github_client is None:
        if Github is None or Auth is None:
            raise EnvironmentError(
                "PyGithub is required for mining. Install requirements.txt "
                "or run: pip install PyGithub"
            )
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN environment variable is required. "
                "Set it before running this script."
            )
        github_client = Github(auth=Auth.Token(token), per_page=100)

    # Print remaining rate limit
    try:
        rate_limit = github_client.get_rate_limit()
        remaining = rate_limit.core.remaining
        limit = rate_limit.core.limit
    except Exception:
        remaining = "unknown"
        limit = "unknown"

    print(f"\n{'='*60}")
    print(f"Mining: {repo_name}")
    print(f"Rate limit remaining: {remaining}/{limit}")
    print(f"{'='*60}")

    repo = _api_call_with_retry(github_client.get_repo, repo_name)
    repo_name_safe = repo_name.replace("/", "_")

    # ---- Load existing partial results -------------------------------------
    raw_path = RAW_DIR / f"{repo_name_safe}.json"
    results = _load_existing_results(raw_path)
    seen_ids: set = {r["id"] for r in results}

    # ---- Collect issues ----------------------------------------------------
    # Strategy: search with bug label first, then keyword search
    issues_to_process: list = []

    # 1. Bug-labelled issues
    try:
        bug_issues = _api_call_with_retry(
            repo.get_issues, state="closed", labels=["bug"],
            sort="updated", direction="desc",
        )
        count = 0
        for issue in bug_issues:
            issues_to_process.append(issue)
            count += 1
            if count >= max_issues:
                break
    except GithubException as exc:
        logger.warning("Could not fetch bug-labelled issues: %s", exc)

    # 2. Keyword search via GitHub search API
    for keyword in ERROR_KEYWORDS:
        if len(issues_to_process) >= max_issues:
            break
        query = f'repo:{repo_name} is:issue "{keyword}"'
        try:
            search_results = _api_call_with_retry(
                github_client.search_issues, query,
            )
            count = 0
            for issue in search_results:
                if issue not in issues_to_process:
                    issues_to_process.append(issue)
                count += 1
                if count >= 50:  # cap per keyword
                    break
        except GithubException as exc:
            logger.warning("Search for '%s' failed: %s", keyword, exc)

    logger.info("Collected %d candidate issues for %s", len(issues_to_process), repo_name)

    # ---- Process each issue ------------------------------------------------
    for idx, issue in enumerate(issues_to_process):
        issue_id = f"{repo_name_safe}-issue-{issue.number}"

        # Skip already-processed issues (resume support)
        if issue_id in seen_ids:
            continue

        if (idx + 1) % 10 == 0:
            try:
                remaining = github_client.get_rate_limit().core.remaining
            except Exception:
                remaining = "unknown"
            print(f"  [{idx+1}/{len(issues_to_process)}] "
                  f"Processing issue #{issue.number}  "
                  f"(rate limit: {remaining})")

        # 1. Extract log snippet
        body = issue.body or ""
        log_snippet = _extract_log_snippet(body)
        if not log_snippet or len(log_snippet) < 20:
            continue

        # 2. Find fixing commit and relevant files
        try:
            relevant_files = _find_fixing_commit(repo, issue)
        except Exception as exc:
            logger.debug("Error finding fix for #%d: %s", issue.number, exc)
            continue

        if not relevant_files:
            continue

        # 3. Apply filters
        if len(relevant_files) > 10:
            logger.debug("Skipping #%d: %d files modified (too many)",
                         issue.number, len(relevant_files))
            continue

        # 4. Build result record
        result: Dict[str, Any] = {
            "id": issue_id,
            "log": log_snippet,
            "relevant_files": relevant_files,
            "error_type": classify_error_type(log_snippet),
            "source": "github",
            "repo": repo_name,
            "issue_url": issue.html_url,
        }
        results.append(result)
        seen_ids.add(issue_id)

        # 5. Save incrementally
        _save_results(results, raw_path)
        logger.info("  ✓ Issue #%d → %d files  [total: %d]",
                     issue.number, len(relevant_files), len(results))

    print(f"  Finished {repo_name}: {len(results)} labeled pairs")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Mine all target repos and merge results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        nargs="+",
        default=TARGET_REPOS,
        help="GitHub repositories to mine, e.g. llvm/llvm-project verilator/verilator.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=200,
        help="Maximum issues to inspect per repository.",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for repo_name in args.repos:
        try:
            pairs = mine_repo(repo_name, max_issues=args.max_issues)
            all_results.extend(pairs)
        except Exception as exc:
            logger.error("Failed to mine %s: %s", repo_name, exc)

    # Merge all results into processed output
    merged_path = PROCESSED_DIR / "github_pairs.json"
    _save_results(all_results, merged_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"Mining complete!")
    print(f"Total labeled pairs: {len(all_results)}")
    print(f"Merged output: {merged_path}")
    print(f"{'='*60}")

    # Per-repo breakdown
    from collections import Counter
    repo_counts = Counter(r["repo"] for r in all_results)
    type_counts = Counter(r["error_type"] for r in all_results)

    print("\nPer-repo breakdown:")
    for repo, count in repo_counts.most_common():
        print(f"  {repo}: {count}")

    print("\nPer-error-type breakdown:")
    for etype, count in type_counts.most_common():
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
