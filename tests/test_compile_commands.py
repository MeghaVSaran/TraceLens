"""Tests for compilation database loading and lookup."""

import json

from src.analysis.compile_commands import CompileCommandsIndex, discover_compile_commands


def test_discover_compile_commands_prefers_build_dir(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    build_dir = repo_root / "build-debug"
    build_dir.mkdir()
    target = build_dir / "compile_commands.json"
    target.write_text("[]", encoding="utf-8")

    found = discover_compile_commands(repo_root, build_dir=build_dir)
    assert found == target.resolve()


def test_compile_commands_extract_include_dirs_and_std_flag(tmp_path):
    repo_root = tmp_path / "repo"
    source_dir = repo_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "foo.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    db_path = repo_root / "compile_commands.json"
    db_path.write_text(
        json.dumps([
            {
                "directory": str(repo_root),
                "file": "src/foo.cpp",
                "command": "clang++ -Iinclude -I third_party/include -DDEBUG -std=c++20 -c src/foo.cpp",
            }
        ]),
        encoding="utf-8",
    )

    index = CompileCommandsIndex.from_file(db_path, repo_root=repo_root)
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.file_path == "src/foo.cpp"
    assert entry.include_dirs == ("include", "third_party/include")
    assert entry.defines == ("DEBUG",)
    assert entry.std_flag == "-std=c++20"


def test_find_best_entry_matches_suffix_and_basename(tmp_path):
    repo_root = tmp_path / "repo"
    foo_dir = repo_root / "absl" / "strings"
    foo_dir.mkdir(parents=True)
    (foo_dir / "str_cat.cc").write_text("void StrCat() {}\n", encoding="utf-8")
    db_path = repo_root / "compile_commands.json"
    db_path.write_text(
        json.dumps([
            {
                "directory": str(repo_root),
                "file": "absl/strings/str_cat.cc",
                "command": "clang++ -c absl/strings/str_cat.cc",
            }
        ]),
        encoding="utf-8",
    )
    index = CompileCommandsIndex.from_file(db_path, repo_root=repo_root)

    assert index.find_best_entry(["absl/strings/str_cat.cc"]).file_path == "absl/strings/str_cat.cc"
    assert index.find_best_entry(["strings/str_cat.cc"]).file_path == "absl/strings/str_cat.cc"
    assert index.find_best_entry(["str_cat.cc"]).file_path == "absl/strings/str_cat.cc"
