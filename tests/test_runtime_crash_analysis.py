"""Runtime sanitizer parsing and evidence-ranking regression tests."""

from src.analysis.crash_analyzer import analyze_crash
from src.ingestion.log_parser import parse_log


REALISTIC_UAF = """
==42==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010
READ of size 4 at 0x602000000010 thread T0
    #0 0x401234 in BufferOwner::read() /repo/runtime_fixture.cpp:15:16
    #1 0x401300 in main /repo/runtime_fixture.cpp:26:22
freed by thread T0 here:
    #0 0x7ffff in operator delete[](void*) ../../../../libsanitizer/asan/asan_new_delete.cpp:155
    #1 0x401180 in BufferOwner::clear() /repo/runtime_fixture.cpp:11:9
    #2 0x4012f0 in main /repo/runtime_fixture.cpp:25:16
previously allocated by thread T0 here:
    #0 0x7fffe in operator new[](unsigned long) ../../../../libsanitizer/asan/asan_new_delete.cpp:98
    #1 0x401100 in BufferOwner::BufferOwner() /repo/runtime_fixture.cpp:7:36
    #2 0x4012e0 in main /repo/runtime_fixture.cpp:24:17
"""


def test_realistic_asan_prefers_user_free_frame_over_runtime_allocator():
    report = analyze_crash(parse_log(REALISTIC_UAF), [])

    assert report.crash_type == "heap-use-after-free"
    assert report.crash_frame.function == "BufferOwner::read"
    assert report.freed_by_frames[0].function == "operator delete[]"
    assert report.freed_by_frames[0].evidence_source == "freed"
    assert report.suspicious_frames[0].function == "BufferOwner::clear"
    assert report.suspicious_frames[0].evidence_source == "freed"
    assert report.confidence == "high"
    assert "Check BufferOwner::clear - this is where the object was freed" in report.suggested_checks


def test_ubsan_is_classified_by_log_parser_and_crash_analyzer():
    log = """
/repo/overflow.cpp:8:18: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
    #0 0x401111 in checkedAdd(int) /repo/overflow.cpp:8:18
    #1 0x401150 in main /repo/overflow.cpp:12:22
"""

    parsed = parse_log(log)
    report = analyze_crash(parsed, [])

    assert parsed.error_type == "ubsan_error"
    assert report.crash_type == "ubsan_error"
    assert report.crash_frame.function == "checkedAdd"
    assert any("undefined operation" in check for check in report.suggested_checks)


def test_leak_sanitizer_is_classified_separately():
    log = """
==7==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 16 byte(s) in 1 object(s) allocated from:
    #0 0x7ffff in operator new[](unsigned long) ../../../../libsanitizer/asan/asan_new_delete.cpp:98
    #1 0x401100 in BufferOwner::allocate() /repo/leak.cpp:7:20
"""

    parsed = parse_log(log)
    report = analyze_crash(parsed, [])

    assert parsed.error_type == "memory_leak"
    assert report.crash_type == "memory_leak"
    assert any("ownership exit path" in check for check in report.suggested_checks)


def test_partial_unsymbolized_frame_is_preserved():
    log = """
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x00007ffff7abc123 in ?? ()
"""

    report = analyze_crash(parse_log(log), [])

    assert report.crash_type == "segfault"
    assert report.crash_frame is not None
    assert report.crash_frame.function == "unknown"
    assert report.crash_frame.file_path == ""
    assert report.confidence == "low"
