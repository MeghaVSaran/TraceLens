"""Tests for repository-level symbol declaration/definition lookup."""

from src.analysis.symbol_locator import locate_symbol_sites


def test_locate_symbol_sites_finds_declaration_and_definition(tmp_path):
    repo_root = tmp_path / "repo"
    header = repo_root / "include" / "parser.h"
    source = repo_root / "src" / "parser.cpp"
    header.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)

    header.write_text(
        "class Parser {\n"
        "public:\n"
        "  void resolveSymbol(int x);\n"
        "};\n",
        encoding="utf-8",
    )
    source.write_text(
        '#include "parser.h"\n'
        "void Parser::resolveSymbol(int x) {\n"
        "  (void)x;\n"
        "}\n",
        encoding="utf-8",
    )

    sites = locate_symbol_sites(repo_root, "Parser::resolveSymbol")

    assert sites["declaration"] is not None
    assert sites["definition"] is not None
    assert sites["declaration"].file_path == "include/parser.h"
    assert sites["definition"].file_path == "src/parser.cpp"


def test_locate_symbol_sites_prefers_candidate_files_first(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "a").mkdir(parents=True)
    (repo_root / "b").mkdir(parents=True)
    (repo_root / "a" / "foo.h").write_text("void helper();\n", encoding="utf-8")
    (repo_root / "b" / "foo.cpp").write_text("void helper() {}\n", encoding="utf-8")

    sites = locate_symbol_sites(repo_root, "helper", candidate_files=["b/foo.cpp", "a/foo.h"])

    assert sites["definition"] is not None
    assert sites["declaration"] is not None
