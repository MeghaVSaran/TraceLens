"""Deterministic Phase 2 diagnosis built on retrieval + build metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Iterable, Optional

from src.analysis.cmake_index import CMakeProjectIndex
from src.analysis.compile_commands import CompileCommandsIndex
from src.analysis.symbol_locator import locate_symbol_sites
from src.retrieval.hybrid_retriever import RetrievalResult


HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
COMMON_EXTERNAL_LINKER_SYMBOLS = {
    "ceil", "ceilf", "ceill", "floor", "floorf", "floorl", "round", "roundf",
    "sqrt", "sqrtf", "sqrtl", "pow", "powf", "powl", "sin", "sinf", "cos",
    "cosf", "tan", "tanf", "exp", "expf", "log", "logf", "fabs", "fabsf",
    "fmod", "fmodf", "trunc", "truncf", "nearbyint", "nearbyintf", "pthread_create",
    "pthread_join", "pthread_mutex_lock", "pthread_mutex_unlock", "dlopen",
    "dlsym", "dlclose", "clock_gettime", "backtrace", "backtrace_symbols",
    "malloc", "calloc", "realloc", "free",
}
EXTERNAL_LINKER_PREFIXES = (
    "__cxa_", "__gxx_", "__libc_", "_z", "pthread_", "std::", "operator new",
    "operator delete", "dl", "clock_", "sin", "cos", "tan", "sqrt", "ceil",
    "floor", "round", "pow", "exp", "log", "fabs", "fmod", "trunc",
)


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
        cmake_index: Optional[CMakeProjectIndex] = None,
    ):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.compile_commands = compile_commands
        self.cmake_index = cmake_index

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
        if error_type in {"segfault", "asan_error", "ubsan_error", "memory_leak"}:
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
        failing_target, provider_targets = self._infer_link_targets(
            parsed_log,
            compile_entry,
            definition_result=definition,
            definition_site=def_site,
        )
        symbol_is_external = self._is_external_linker_symbol(symbol, decl_site, def_site)
        target_reaches_provider = self._target_links_any(failing_target, provider_targets)
        summary = f"Linker failure around {symbol or 'an unresolved symbol'}."

        if symbol_is_external:
            likely_cause = (
                f"{symbol or 'The unresolved symbol'} looks like an external runtime or system-library symbol rather than a repo-owned C++ symbol."
            )
            suggested_fix = (
                "Check linker flags and system-library dependencies for the failing target, such as math, pthread, dl, or C++ runtime linkage."
            )
            confidence = "high"
        elif failing_target and provider_targets and not target_reaches_provider:
            provider_names = ", ".join(provider.name for provider in provider_targets)
            likely_cause = (
                f"{symbol or 'The symbol'} resolves to provider target(s) {provider_names}, but the failing target "
                f"{failing_target.name} does not appear to link them."
            )
            suggested_fix = (
                f"Add the provider target to target_link_libraries({failing_target.name} ...) or otherwise route its object/library into the link step."
            )
            confidence = "high"
        elif failing_target and provider_targets and target_reaches_provider:
            provider_names = ", ".join(provider.name for provider in provider_targets)
            likely_cause = (
                f"{symbol or 'The symbol'} resolves to provider target(s) {provider_names}, and the current CMake graph shows "
                f"{failing_target.name} already reaches them. This points away from a simple missing target_link_libraries edge."
            )
            suggested_fix = (
                "Compare the failing build's actual link command and repo revision against the indexed source; check conditional CMake options, stale build files, and whether the failing environment uses an older target graph."
            )
            confidence = "medium"
        elif decl_site and def_site:
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
        if symbol_is_external:
            evidence.append("Symbol classified as external/system-library style.")
        if not symbol_is_external and decl_site:
            evidence.append(f"Repo declaration site: {decl_site.file_path}:{decl_site.line_number}")
        elif not symbol_is_external and decl:
            evidence.append(f"Header candidate: {decl.file_path}:{decl.start_line}")
        if not symbol_is_external and def_site:
            evidence.append(f"Repo definition site: {def_site.file_path}:{def_site.line_number}")
        elif not symbol_is_external and definition:
            evidence.append(f"Definition candidate: {definition.file_path}:{definition.start_line}")
        if failing_target:
            evidence.append(f"Failing source appears in target: {failing_target.name}")
        elif getattr(parsed_log, "build_targets", []):
            evidence.append("Build target hint(s) from log: " + ", ".join(parsed_log.build_targets))
        if not symbol_is_external and provider_targets:
            evidence.append(
                "Definition provider target(s): " + ", ".join(target.name for target in provider_targets)
            )
            if failing_target:
                relation = "reaches" if target_reaches_provider else "does not reach"
                evidence.append(
                    f"Build graph: {failing_target.name} {relation} provider target(s)."
                )

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
        failure_kind = self._compiler_failure_kind(parsed_log, symbol)
        location = f"{best.file_path}:{best.start_line}" if best else "the top-ranked code context"

        if failure_kind == "platform_macro":
            likely_cause = (
                f"{symbol or 'The identifier'} looks like a platform or feature-test macro, not a repo-owned C++ symbol. "
                f"The failure is likely a portability/configuration issue near {location}, not a normal C++ symbol-lookup problem."
            )
            suggested_fix = (
                "Check the platform header, feature-test macros, and OS guards used by the failing target; add a portable fallback if the macro is unavailable."
            )
            confidence = "medium" if symbol else "low"
        elif failure_kind == "constructor_signature":
            likely_cause = (
                f"The constructor call for {symbol or 'this type'} does not match the available overloads. "
                f"Retrieval points to {location}, so the bug is likely an argument type, callable signature, or lifetime mismatch at the call site."
            )
            suggested_fix = (
                "Compare the failing constructor arguments against the retrieved declaration and adjust the callable, wrapper type, or explicit construction."
            )
            confidence = "high" if best and symbol else "medium"
        elif failure_kind == "call_signature":
            likely_cause = (
                f"The compiler found a call to {symbol or 'a function'}, but no visible overload accepts the argument types in the error. "
                f"Retrieval points to {location}, so this is more likely an overload/type mismatch than a missing file."
            )
            suggested_fix = (
                "Compare the failing call's argument types with the retrieved overloads; add an explicit cast/template argument or choose the intended overload."
            )
            confidence = "high" if best and symbol else "medium"
        elif failure_kind == "member_lookup":
            likely_cause = (
                f"The member lookup for {symbol or 'the requested member'} failed. "
                f"Retrieval points to {location}, so the receiver type may differ from the type that actually owns that member."
            )
            suggested_fix = "Check the receiver type, namespace/using context, and whether a feature flag or version mismatch removed the member."
            confidence = "high" if best and symbol else "medium"
        elif failure_kind == "type_visibility":
            likely_cause = (
                f"{symbol or 'The type'} is referenced before the compiler has a complete visible declaration. "
                f"Retrieval points to {location}, so this is likely include-order, forward-declaration, or conditional-compilation related."
            )
            suggested_fix = "Include the owning header at the use site or move the operation to a location where the full type definition is visible."
            confidence = "high" if best and symbol else "medium"
        elif best:
            likely_cause = (
                f"{symbol or 'The symbol'} is most likely owned or declared near {best.file_path}:{best.start_line}. "
                "The compile error is likely coming from missing visibility, wrong namespace, or incompatible call-site usage."
            )
            suggested_fix = "Compare the failing use with the top-ranked declaration or implementation and check include, namespace, and signature details."
            confidence = "high" if symbol else "medium"
        else:
            likely_cause = (
                "The parser extracted a compiler failure, but retrieval evidence is still broad rather than symbol-specific."
            )
            suggested_fix = "Inspect the top-ranked files and rerun with a fuller compiler diagnostic if available."
            confidence = "low"

        evidence = self._common_evidence(parsed_log, results, compile_entry)
        if symbol:
            evidence.insert(0, f"Compiler symbol: {symbol}")
        evidence.append(f"Compiler failure pattern: {failure_kind}")
        if getattr(parsed_log, "build_targets", []):
            evidence.append("Build target hint(s) from log: " + ", ".join(parsed_log.build_targets))
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

    def _infer_link_targets(self, parsed_log, compile_entry, definition_result, definition_site):
        """Infer failing and provider targets from compile/CMake metadata."""
        if not self.cmake_index:
            return None, []
        failing_target = None
        provider_targets = []

        for hint in getattr(parsed_log, "build_targets", []):
            hinted_targets = self.cmake_index.find_targets_by_hint(hint)
            if hinted_targets:
                failing_target = hinted_targets[0]
                break

        if failing_target is None and compile_entry is not None:
            candidates = self.cmake_index.get_targets_for_file(compile_entry.file_path)
            if candidates:
                failing_target = candidates[0]

        provider_file = ""
        if definition_site is not None:
            provider_file = definition_site.file_path
        elif definition_result is not None:
            provider_file = definition_result.file_path
        if provider_file:
            provider_targets = self.cmake_index.get_targets_for_file(provider_file)

        return failing_target, provider_targets

    def _target_links_any(self, target, providers) -> bool:
        """Return True if a target already links any provider target name."""
        if target is None or not providers:
            return False
        if self.cmake_index:
            return self.cmake_index.target_reaches_any(target, providers)
        linked = set(target.links)
        return any(provider.name in linked for provider in providers)

    def _is_external_linker_symbol(self, symbol: str, decl_site, def_site) -> bool:
        """Heuristic classifier for system/runtime linker symbols."""
        value = (symbol or "").strip()
        if not value:
            return False
        normalized = value.lower()
        if normalized in COMMON_EXTERNAL_LINKER_SYMBOLS:
            return True
        if any(normalized.startswith(prefix.lower()) for prefix in EXTERNAL_LINKER_PREFIXES):
            return True
        if decl_site is not None or def_site is not None:
            return False
        return bool(
            "::" not in value
            and len(value) <= 32
            and value.replace("_", "").isalnum()
            and value == value.lower()
        )

    def _compiler_failure_kind(self, parsed_log, symbol: str) -> str:
        """Classify compiler diagnostics into fix-oriented buckets."""
        text = f"{getattr(parsed_log, 'error_message', '')}\n{getattr(parsed_log, 'raw_log', '')}".lower()
        symbol_value = symbol or ""

        if self._looks_like_macro(symbol_value) and (
            "undeclared identifier" in text
            or "was not declared" in text
            or "has not been declared" in text
        ):
            return "platform_macro"
        if "no matching function for call" in text or "no matching constructor" in text:
            if self._looks_like_constructor_symbol(symbol_value) or "constructor" in text:
                return "constructor_signature"
            return "call_signature"
        if "has no member named" in text or "is not a member of" in text:
            return "member_lookup"
        if "does not name a type" in text or "invalid use of incomplete type" in text:
            return "type_visibility"
        if "no declaration matches" in text:
            return "call_signature"
        return "generic"

    def _looks_like_macro(self, symbol: str) -> bool:
        """Return True for C-style macro/config identifiers such as MAP_ANONYMOUS."""
        value = (symbol or "").strip()
        return bool(
            value
            and value.upper() == value
            and "_" in value
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", value)
        )

    def _looks_like_constructor_symbol(self, symbol: str) -> bool:
        """Detect qualified constructor names like absl::Condition::Condition."""
        parts = [part for part in (symbol or "").split("::") if part]
        if len(parts) < 2:
            return False
        return parts[-1].split("<", 1)[0] == parts[-2].split("<", 1)[0]
