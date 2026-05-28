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
