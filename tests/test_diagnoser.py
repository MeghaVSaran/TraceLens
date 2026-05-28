"""Tests for deterministic Phase 2 diagnosis generation."""

import json

from src.analysis.cmake_index import CMakeProjectIndex
from src.analysis.compile_commands import CompileCommandsIndex
from src.analysis.diagnoser import FailureDiagnoser
from src.ingestion.log_parser import parse_log
from src.retrieval.hybrid_retriever import RetrievalResult


def _result(file_path, function_name, start_line=1, score=0.9, symbol_name=""):
    return RetrievalResult(
        rank=1,
        chunk_id=f"{file_path}::{function_name}",
        file_path=file_path,
        function_name=function_name,
        start_line=start_line,
        score=score,
        dense_score=score,
        bm25_score=score,
        symbol_score=0.0,
        symbol_name=symbol_name,
    )


def test_linker_diagnosis_mentions_declaration_and_definition(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    include_dir = repo_root / "absl" / "strings"
    src_dir = repo_root / "src"
    include_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    (include_dir / "str_cat.h").write_text(
        "namespace absl { std::string StrCat(); }\n",
        encoding="utf-8",
    )
    (include_dir / "str_cat.cc").write_text(
        "namespace absl { std::string StrCat() { return {}; } }\n",
        encoding="utf-8",
    )
    (src_dir / "use_str_cat.cpp").write_text(
        '#include "absl/strings/str_cat.h"\n'
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(absl_strings absl/strings/str_cat.cc)\n"
        "add_executable(app src/use_str_cat.cpp)\n",
        encoding="utf-8",
    )
    db_path = repo_root / "compile_commands.json"
    db_path.write_text(
        json.dumps([
            {
                "directory": str(repo_root),
                "file": "src/use_str_cat.cpp",
                "command": "clang++ -std=c++20 -c src/use_str_cat.cpp",
            }
        ]),
        encoding="utf-8",
    )
    compile_commands = CompileCommandsIndex.from_file(db_path, repo_root=repo_root)
    cmake_index = CMakeProjectIndex.from_repo(repo_root)
    diagnoser = FailureDiagnoser(
        repo_root=repo_root,
        compile_commands=compile_commands,
        cmake_index=cmake_index,
    )
    parsed = parse_log(
        "/usr/bin/ld: undefined reference to `absl::StrCat`\n"
        "src/use_str_cat.cpp:(.text+0x1): undefined reference\n"
    )
    parsed.source_paths = ["src/use_str_cat.cpp"]
    results = [
        _result("absl/strings/str_cat.h", "absl::StrCat", 12, 0.87, "StrCat"),
        _result("absl/strings/str_cat.cc", "absl::StrCat", 48, 0.92, "StrCat"),
    ]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert diagnosis.error_type == "linker_error"
    assert diagnosis.confidence == "high"
    assert "does not appear to link" in diagnosis.likely_cause
    assert "absl/strings/str_cat.cc" in " ".join(diagnosis.evidence)
    assert any("Repo declaration site" in item for item in diagnosis.evidence)
    assert any("Failing source appears in target: app" in item for item in diagnosis.evidence)
    assert any("Definition provider target(s): absl_strings" in item for item in diagnosis.evidence)


def test_include_diagnosis_uses_compile_command_context(tmp_path):
    repo_root = tmp_path / "repo"
    header_dir = repo_root / "include" / "parser"
    source_dir = repo_root / "src"
    header_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    (header_dir / "resolve.h").write_text("// header\n", encoding="utf-8")
    (source_dir / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    db_path = repo_root / "compile_commands.json"
    db_path.write_text(
        json.dumps([
            {
                "directory": str(repo_root),
                "file": "src/main.cpp",
                "command": "clang++ -Iinclude -std=c++20 -c src/main.cpp",
            }
        ]),
        encoding="utf-8",
    )
    compile_commands = CompileCommandsIndex.from_file(db_path, repo_root=repo_root)
    diagnoser = FailureDiagnoser(repo_root=repo_root, compile_commands=compile_commands)
    parsed = parse_log(
        "src/main.cpp:4:10: fatal error: parser/resolve.h: No such file or directory\n"
    )
    parsed.source_paths = ["src/main.cpp"]
    results = [_result("src/main.cpp", "main", 1, 0.8)]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert diagnosis.error_type == "include_error"
    assert diagnosis.compile_command_file == "src/main.cpp"
    assert "parser/resolve.h" in diagnosis.likely_cause
    assert any("Compile command matched" in item for item in diagnosis.evidence)


def test_memory_diagnosis_surfaces_stack_evidence():
    diagnoser = FailureDiagnoser()
    parsed = parse_log(
        "Program received signal SIGSEGV, Segmentation fault.\n"
        "#0  0x01 in Parser::resolveSymbol at parser.cpp:42\n"
        "#1  0x02 in SymbolTable::lookup at table.cpp:99\n"
    )
    results = [_result("parser.cpp", "Parser::resolveSymbol", 42, 0.88, "resolveSymbol")]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert diagnosis.error_type == "segfault"
    assert diagnosis.confidence == "medium"
    assert any("Stack frame:" in item for item in diagnosis.evidence)


def test_external_linker_symbol_is_treated_as_system_library_issue():
    diagnoser = FailureDiagnoser()
    parsed = parse_log(
        "/usr/bin/ld: low_level_alloc_test.cc.o: undefined reference to `ceilf'\n"
    )
    results = [
        _result("absl/base/internal/low_level_alloc.h", "__file__", 1, 0.95),
        _result("absl/synchronization/mutex.cc", "ScopedDeadlockReportBuffers", 1365, 0.58),
    ]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert diagnosis.error_type == "linker_error"
    assert "external runtime or system-library symbol" in diagnosis.likely_cause
    assert any("external/system-library" in item for item in diagnosis.evidence)


def test_linker_target_hint_is_used_when_compile_entry_mapping_is_missing(tmp_path):
    repo_root = tmp_path / "repo"
    provider_dir = repo_root / "absl" / "flags"
    consumer_dir = provider_dir / "internal"
    provider_dir.mkdir(parents=True)
    consumer_dir.mkdir(parents=True)
    (provider_dir / "commandlineflag.cc").write_text(
        "bool CommandLineFlag::IsRetired() const { return false; }\n",
        encoding="utf-8",
    )
    (provider_dir / "commandlineflag.h").write_text(
        "class CommandLineFlag { public: bool IsRetired() const; };\n",
        encoding="utf-8",
    )
    (consumer_dir / "flag.cc").write_text("void use_flag() {}\n", encoding="utf-8")
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(absl_flags_internal absl/flags/internal/flag.cc)\n"
        "add_library(absl_flags absl/flags/commandlineflag.cc)\n",
        encoding="utf-8",
    )
    cmake_index = CMakeProjectIndex.from_repo(repo_root)
    diagnoser = FailureDiagnoser(repo_root=repo_root, cmake_index=cmake_index)
    parsed = parse_log(
        "/usr/bin/ld: CMakeFiles/flags_internal.dir/internal/flag.cc.o:"
        " undefined reference to `absl::CommandLineFlag::IsRetired() const'\n",
        repo_root=repo_root,
    )
    results = [
        _result("absl/flags/commandlineflag.cc", "CommandLineFlag::IsRetired", 1, 1.0, "IsRetired"),
        _result("absl/flags/internal/flag.cc", "__file__", 1, 0.95),
    ]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert any("Failing source appears in target: absl_flags_internal" in item for item in diagnosis.evidence)
    assert any("Definition provider target(s): absl_flags" in item for item in diagnosis.evidence)


def test_linker_diagnosis_does_not_claim_missing_edge_when_provider_is_transitive(tmp_path):
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "provider.cpp").write_text("int provide() { return 1; }\n", encoding="utf-8")
    (src_dir / "mid.cpp").write_text("int mid() { return provide(); }\n", encoding="utf-8")
    (src_dir / "consumer.cpp").write_text("int main() { return mid(); }\n", encoding="utf-8")
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(provider src/provider.cpp)\n"
        "add_library(mid src/mid.cpp)\n"
        "target_link_libraries(mid PRIVATE provider)\n"
        "add_executable(app src/consumer.cpp)\n"
        "target_link_libraries(app PRIVATE mid)\n",
        encoding="utf-8",
    )
    db_path = repo_root / "compile_commands.json"
    db_path.write_text(
        json.dumps([
            {
                "directory": str(repo_root),
                "file": "src/consumer.cpp",
                "command": "clang++ -std=c++20 -c src/consumer.cpp",
            }
        ]),
        encoding="utf-8",
    )
    compile_commands = CompileCommandsIndex.from_file(db_path, repo_root=repo_root)
    cmake_index = CMakeProjectIndex.from_repo(repo_root)
    diagnoser = FailureDiagnoser(
        repo_root=repo_root,
        compile_commands=compile_commands,
        cmake_index=cmake_index,
    )
    parsed = parse_log(
        "/usr/bin/ld: undefined reference to `provide`\n"
        "src/consumer.cpp:(.text+0x1): undefined reference\n"
    )
    parsed.source_paths = ["src/consumer.cpp"]
    results = [
        _result("src/provider.cpp", "provide", 1, 0.97, "provide"),
        _result("src/consumer.cpp", "main", 1, 0.80, "main"),
    ]

    diagnosis = diagnoser.diagnose(parsed, results)

    assert "does not appear to link" not in diagnosis.likely_cause
    assert any("Definition provider target(s): provider" in item for item in diagnosis.evidence)
