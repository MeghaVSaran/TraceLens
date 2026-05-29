"""Offline tests for GitHub issue mining helpers."""

from scripts.mine_github_issues import TARGET_REPOS, classify_error_type


def test_miner_targets_include_eda_relevant_repos():
    assert "verilator/verilator" in TARGET_REPOS
    assert "YosysHQ/yosys" in TARGET_REPOS
    assert "The-OpenROAD-Project/OpenROAD" in TARGET_REPOS


def test_classify_error_type_handles_more_realistic_linker_logs():
    assert classify_error_type("ld: undefined symbol: Foo::bar") == "linker_error"
    assert classify_error_type("'Foo::bar()' used but never defined") == "linker_error"


def test_classify_error_type_handles_sanitizer_and_build_logs():
    assert classify_error_type("AddressSanitizer: heap-buffer-overflow on address") == "asan_error"
    assert classify_error_type("No rule to make target `generated/foo.h`") == "build_system_error"
