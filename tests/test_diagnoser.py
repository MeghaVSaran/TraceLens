"""Tests for deterministic Phase 2 diagnosis generation."""

import json

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
    diagnoser = FailureDiagnoser(repo_root=repo_root, compile_commands=compile_commands)
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
    assert "declared" in diagnosis.likely_cause
    assert "absl/strings/str_cat.cc" in " ".join(diagnosis.evidence)


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
