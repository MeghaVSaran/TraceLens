"""Tests for lightweight CMake target/source/dependency indexing."""

from src.analysis.cmake_index import CMakeProjectIndex


def test_cmake_index_parses_targets_and_links(tmp_path):
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "provider.cpp").write_text("void provide() {}\n", encoding="utf-8")
    (src_dir / "consumer.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(provider src/provider.cpp)\n"
        "add_executable(app src/consumer.cpp)\n"
        "target_link_libraries(app PRIVATE provider)\n",
        encoding="utf-8",
    )

    index = CMakeProjectIndex.from_repo(repo_root)

    provider_targets = index.get_targets_for_file("src/provider.cpp")
    app_targets = index.get_targets_for_file("src/consumer.cpp")
    assert provider_targets[0].name == "provider"
    assert app_targets[0].name == "app"
    assert "provider" in app_targets[0].links


def test_cmake_index_handles_subdirectory_cmakelists(tmp_path):
    repo_root = tmp_path / "repo"
    module_dir = repo_root / "module"
    module_dir.mkdir(parents=True)
    (module_dir / "worker.cc").write_text("void work() {}\n", encoding="utf-8")
    (module_dir / "CMakeLists.txt").write_text(
        "add_library(worker worker.cc)\n",
        encoding="utf-8",
    )

    index = CMakeProjectIndex.from_repo(repo_root)

    targets = index.get_targets_for_file("module/worker.cc")
    assert len(targets) == 1
    assert targets[0].defined_in == "module/CMakeLists.txt"


def test_cmake_index_finds_target_by_hint_suffix(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(absl_flags_internal flag.cc)\n",
        encoding="utf-8",
    )
    (repo_root / "flag.cc").write_text("void x() {}\n", encoding="utf-8")

    index = CMakeProjectIndex.from_repo(repo_root)
    targets = index.find_targets_by_hint("flags_internal")

    assert len(targets) == 1
    assert targets[0].name == "absl_flags_internal"


def test_cmake_index_parses_absl_cc_library_blocks(tmp_path):
    repo_root = tmp_path / "repo"
    flags_dir = repo_root / "absl" / "flags"
    flags_dir.mkdir(parents=True)
    (flags_dir / "commandlineflag.cc").write_text("bool x() { return false; }\n", encoding="utf-8")
    (flags_dir / "commandlineflag.h").write_text("bool x();\n", encoding="utf-8")
    (flags_dir / "CMakeLists.txt").write_text(
        "absl_cc_library(\n"
        "  NAME\n"
        "    flags_commandlineflag\n"
        "  SRCS\n"
        "    \"commandlineflag.cc\"\n"
        "  HDRS\n"
        "    \"commandlineflag.h\"\n"
        "  DEPS\n"
        "    absl::config\n"
        "    absl::strings\n"
        ")\n",
        encoding="utf-8",
    )

    index = CMakeProjectIndex.from_repo(repo_root)

    targets = index.get_targets_for_file("absl/flags/commandlineflag.cc")
    assert len(targets) == 1
    assert targets[0].name == "flags_commandlineflag"
    assert "strings" in targets[0].links


def test_cmake_index_parses_project_specific_cc_macro_blocks(tmp_path):
    repo_root = tmp_path / "repo"
    engine_dir = repo_root / "src" / "engine"
    engine_dir.mkdir(parents=True)
    (engine_dir / "solver.cpp").write_text("void solve() {}\n", encoding="utf-8")
    (engine_dir / "solver.h").write_text("void solve();\n", encoding="utf-8")
    (engine_dir / "CMakeLists.txt").write_text(
        "cadence_cc_library(\n"
        "  NAME\n"
        "    timing_solver\n"
        "  SRCS\n"
        "    solver.cpp\n"
        "  HDRS\n"
        "    solver.h\n"
        "  DEPS\n"
        "    core::graph\n"
        ")\n",
        encoding="utf-8",
    )

    index = CMakeProjectIndex.from_repo(repo_root)

    targets = index.get_targets_for_file("src/engine/solver.cpp")
    assert len(targets) == 1
    assert targets[0].name == "timing_solver"
    assert targets[0].kind == "library"
    assert "graph" in targets[0].links


def test_cmake_index_checks_transitive_target_reachability(tmp_path):
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "consumer.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (src_dir / "mid.cpp").write_text("void mid() {}\n", encoding="utf-8")
    (src_dir / "provider.cpp").write_text("void provide() {}\n", encoding="utf-8")
    (repo_root / "CMakeLists.txt").write_text(
        "add_library(provider src/provider.cpp)\n"
        "add_library(mid src/mid.cpp)\n"
        "target_link_libraries(mid PRIVATE provider)\n"
        "add_executable(app src/consumer.cpp)\n"
        "target_link_libraries(app PRIVATE mid)\n",
        encoding="utf-8",
    )

    index = CMakeProjectIndex.from_repo(repo_root)
    app = index.get_target("app")
    provider = index.get_target("provider")

    assert app is not None
    assert provider is not None
    assert index.target_reaches_any(app, [provider]) is True
