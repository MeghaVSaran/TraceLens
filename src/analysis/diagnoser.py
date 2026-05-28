"""Deterministic Phase 2 diagnosis built on retrieval + build metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.analysis.compile_commands import CompileCommandsIndex
from src.analysis.symbol_locator import locate_symbol_sites
from src.retrieval.hybrid_retriever import RetrievalResult


HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


@dataclass
class DiagnosisReport:
    """Structured diagnosis returned by query/watch mode."""

    error_type: str
    summary: str
    likely_cause: str
    suggested_fix: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    compile_command_file: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-friendly dictionary."""
        return asdict(self)


class FailureDiagnoser:
    """Produce build-aware diagnoses from parsed logs and retrieval evidence."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        compile_commands: Optional[CompileCommandsIndex] = None,
    ):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.compile_commands = compile_commands

    def diagnose(self, parsed_log, results: Iterable[RetrievalResult]) -> DiagnosisReport:
        """Create a structured diagnosis for the given failure."""
        results = list(results)
        error_type = getattr(parsed_log, "error_type", "unknown")
        primary_files = self._top_files(results)
        compile_entry = self._lookup_compile_entry(parsed_log)

        if error_type == "include_error":
            return self._diagnose_include(parsed_log, results, primary_files, compile_entry)
        if error_type == "linker_error":
            return self._diagnose_linker(parsed_log, results, primary_files, compile_entry)
        if error_type == "compiler_error":
            return self._diagnose_compiler(parsed_log, results, primary_files, compile_entry)
        if error_type in {"segfault", "asan_error"}:
            return self._diagnose_memory(parsed_log, results, primary_files, compile_entry)
        if error_type == "build_system_error":
            return self._diagnose_build_system(parsed_log, results, primary_files, compile_entry)
        if error_type == "runtime_exception":
            return self._diagnose_runtime(parsed_log, results, primary_files, compile_entry)
        return DiagnosisReport(
            error_type=error_type,
            summary="Retrieved candidate files, but the failure type is still generic.",
            likely_cause="The current parser extracted limited structured evidence from this log.",
            suggested_fix="Inspect the top-ranked files and rerun with a fuller build/runtime log if available.",
            confidence="low",
            evidence=self._common_evidence(parsed_log, results, compile_entry),
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_include(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        header = self._first_file_hint(parsed_log)
        source = compile_entry.file_path if compile_entry else self._best_source_hint(parsed_log)
        in_repo = self._repo_contains(header)
        summary = f"Include failure while compiling {source or 'the current translation unit'}."
        if header and in_repo:
            likely_cause = (
                f"{header} appears to exist in the repo, but the compile command for "
                f"{source or 'this file'} is not making that header reachable."
            )
            suggested_fix = "Check the target's include directories, dependency wiring, and generated-header setup."
            confidence = "high" if compile_entry else "medium"
        else:
            likely_cause = (
                f"{header or 'The missing header'} does not appear reachable from the current build context. "
                "This is often a missing include path, missing dependency, or generated file issue."
            )
            suggested_fix = "Verify the header path, target dependency, and whether the file is generated before compile."
            confidence = "medium" if compile_entry or header else "low"

        evidence = self._common_evidence(parsed_log, results, compile_entry)
        if header:
            evidence.insert(0, f"Missing header: {header}")
        if compile_entry and compile_entry.include_dirs:
            evidence.append(f"Compile command exposes {len(compile_entry.include_dirs)} include path(s).")

        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_linker(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        symbol = self._first_identifier(parsed_log)
        decl = self._find_matching_result(results, symbol, suffixes=HEADER_SUFFIXES)
        definition = self._find_matching_result(results, symbol, suffixes=SOURCE_SUFFIXES)
        repo_sites = self._lookup_symbol_sites(symbol, primary_files)
        decl_site = repo_sites.get("declaration")
        def_site = repo_sites.get("definition")
        summary = f"Linker failure around {symbol or 'an unresolved symbol'}."

        if decl_site and def_site:
            likely_cause = (
                f"{symbol or 'The symbol'} appears declared in {decl_site.file_path} and defined in "
                f"{def_site.file_path}, which strongly suggests the defining object or library is missing from the failing link."
            )
            suggested_fix = (
                "Check the target or library dependencies for the failing binary and ensure the definition file's object is linked."
            )
            confidence = "high"
        elif decl and definition:
            likely_cause = (
                f"{symbol or 'The symbol'} appears declared in {decl.file_path} and implemented in "
                f"{definition.file_path}, which usually means the defining object/library is not linked into the failing target."
            )
            suggested_fix = (
                "Inspect the target or library dependencies for the failing binary and make sure the definition file's target is linked."
            )
            confidence = "high"
        elif definition:
            likely_cause = (
                f"The top retrieval evidence points to {definition.file_path} as the implementation site for "
                f"{symbol or 'the unresolved symbol'}, suggesting a missing object file or library edge."
            )
            suggested_fix = "Check whether the file's library/target is part of the failing link step."
            confidence = "medium"
        else:
            likely_cause = (
                "The log contains unresolved-link evidence, but retrieval did not yet isolate both declaration and definition sites."
            )
            suggested_fix = "Inspect the top-ranked files and compare the symbol's declaration against the actual link inputs."
            confidence = "low"

        evidence = self._common_evidence(parsed_log, results, compile_entry)
        if symbol:
            evidence.insert(0, f"Linker symbol: {symbol}")
        if decl_site:
            evidence.append(f"Repo declaration site: {decl_site.file_path}:{decl_site.line_number}")
        elif decl:
            evidence.append(f"Header candidate: {decl.file_path}:{decl.start_line}")
        if def_site:
            evidence.append(f"Repo definition site: {def_site.file_path}:{def_site.line_number}")
        elif definition:
            evidence.append(f"Definition candidate: {definition.file_path}:{definition.start_line}")

        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_compiler(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        symbol = self._first_identifier(parsed_log)
        best = self._find_matching_result(results, symbol)
        summary = f"Compiler failure around {symbol or 'a missing/invalid symbol'}."

        if best:
            likely_cause = (
                f"{symbol or 'The symbol'} is most likely owned or declared near {best.file_path}:{best.start_line}. "
                "The compile error is likely coming from a missing include, wrong namespace, or signature mismatch."
            )
            confidence = "high" if symbol else "medium"
        else:
            likely_cause = (
                "The parser extracted a compiler failure, but retrieval evidence is still broad rather than symbol-specific."
            )
            confidence = "low"

        suggested_fix = "Compare the failing call/site with the top-ranked declaration or implementation and check include/namespace/signature details."
        evidence = self._common_evidence(parsed_log, results, compile_entry)
        if symbol:
            evidence.insert(0, f"Compiler symbol: {symbol}")
        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_memory(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        frames = [frame for frame in getattr(parsed_log, "stack_frames", [])[:3]]
        best = results[0] if results else None
        summary = "Runtime memory failure triaged from stack evidence."
        if best:
            likely_cause = (
                f"The crash or sanitizer report is clustering around {best.file_path}:{best.start_line}. "
                "The root cause may still be one caller earlier, so the top frames should be inspected in order."
            )
            confidence = "medium"
        else:
            likely_cause = "The log indicates a memory error, but no high-confidence code location was retrieved."
            confidence = "low"
        suggested_fix = "Inspect the top frame and its immediate caller for invalid ownership, nullability, or bounds assumptions."
        evidence = self._common_evidence(parsed_log, results, compile_entry)
        for frame in frames:
            evidence.append(f"Stack frame: {frame}")
        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_build_system(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        subject = self._first_identifier(parsed_log) or "build configuration"
        summary = f"Build-system failure around {subject}."
        likely_cause = (
            "The error looks configuration-level rather than code-local, so the next step is to inspect target wiring, package discovery, and generated build files."
        )
        suggested_fix = "Check package discovery, target names, and generated build artifacts before chasing source-code changes."
        confidence = "medium" if self.compile_commands else "low"
        evidence = self._common_evidence(parsed_log, results, compile_entry)
        if not self.compile_commands:
            evidence.append("No compile_commands.json was found, so build-context evidence is limited.")
        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _diagnose_runtime(self, parsed_log, results, primary_files, compile_entry) -> DiagnosisReport:
        exception_name = self._first_identifier(parsed_log) or "runtime exception"
        summary = f"Runtime exception triaged around {exception_name}."
        likely_cause = (
            "The failure is surfacing as a runtime exception; the top-ranked files are the best candidate throw sites or state owners."
        )
        suggested_fix = "Inspect the throw site, nearby invariants, and the caller that supplied the failing input/state."
        evidence = self._common_evidence(parsed_log, results, compile_entry)
        evidence.insert(0, f"Exception signal: {exception_name}")
        return DiagnosisReport(
            error_type=parsed_log.error_type,
            summary=summary,
            likely_cause=likely_cause,
            suggested_fix=suggested_fix,
            confidence="medium" if results else "low",
            evidence=evidence,
            relevant_files=primary_files,
            compile_command_file=compile_entry.file_path if compile_entry else "",
        )

    def _common_evidence(self, parsed_log, results, compile_entry) -> list[str]:
        evidence: list[str] = []
        if compile_entry is not None:
            evidence.append(f"Compile command matched: {compile_entry.file_path}")
            if compile_entry.std_flag:
                evidence.append(f"Language mode: {compile_entry.std_flag}")
        if results:
            top = results[0]
            evidence.append(
                f"Top retrieval result: {top.file_path}:{top.start_line} ({top.function_name}) score={top.score:.3f}"
            )
        identifiers = list(getattr(parsed_log, "identifiers", []))
        if identifiers:
            evidence.append(f"Identifiers extracted: {', '.join(identifiers[:3])}")
        return evidence

    def _lookup_compile_entry(self, parsed_log):
        if not self.compile_commands:
            return None
        hints = list(getattr(parsed_log, "source_paths", [])) + list(getattr(parsed_log, "file_hints", []))
        return self.compile_commands.find_best_entry(hints)

    def _find_matching_result(
        self,
        results: Iterable[RetrievalResult],
        symbol: str,
        suffixes: Optional[set[str]] = None,
    ) -> Optional[RetrievalResult]:
        symbol = (symbol or "").strip()
        if not symbol:
            return next(iter(results), None)
        symbol_tail = symbol.split("::")[-1].lower()
        for result in results:
            if suffixes and Path(result.file_path).suffix.lower() not in suffixes:
                continue
            fn = (result.function_name or "").lower()
            sym = (getattr(result, "symbol_name", "") or "").lower()
            if symbol.lower() in fn or symbol_tail == sym or symbol_tail in fn:
                return result
        for result in results:
            if suffixes and Path(result.file_path).suffix.lower() not in suffixes:
                continue
            return result
        return None

    def _top_files(self, results: Iterable[RetrievalResult], limit: int = 3) -> list[str]:
        files = []
        for result in results:
            if result.file_path not in files:
                files.append(result.file_path)
            if len(files) >= limit:
                break
        return files

    def _first_identifier(self, parsed_log) -> str:
        identifiers = list(getattr(parsed_log, "identifiers", []))
        return identifiers[0] if identifiers else ""

    def _first_file_hint(self, parsed_log) -> str:
        hints = list(getattr(parsed_log, "file_hints", []))
        return hints[0] if hints else ""

    def _best_source_hint(self, parsed_log) -> str:
        for hint in list(getattr(parsed_log, "source_paths", [])) + list(getattr(parsed_log, "file_hints", [])):
            suffix = Path(str(hint)).suffix.lower()
            if suffix in SOURCE_SUFFIXES:
                return str(hint)
        return ""

    def _repo_contains(self, hint: str) -> bool:
        if not hint or self.repo_root is None:
            return False
        candidate = (self.repo_root / hint).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return False
        return candidate.exists()

    def _lookup_symbol_sites(self, symbol: str, candidate_files: list[str]) -> dict:
        """Resolve declaration/definition sites using a repo scan when possible."""
        if self.repo_root is None or not symbol:
            return {"declaration": None, "definition": None}
        return locate_symbol_sites(
            self.repo_root,
            symbol,
            candidate_files=candidate_files,
        )
