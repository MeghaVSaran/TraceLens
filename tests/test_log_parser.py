"""
Tests for src/ingestion/log_parser.py

Covers all 8 error categories, multi-line logs, unknown errors,
multiple-error-line priority, deeply-namespaced identifiers,
expanded compiler patterns, __BOGUS_ stripping, and source-path extraction.
"""

import pytest
import tempfile
from pathlib import Path
from src.ingestion.log_parser import (
    ParsedLog,
    parse_log,
    extract_error_type,
    extract_identifiers,
    extract_file_hints,
    extract_build_targets,
    extract_source_paths,
)


# ---------------------------------------------------------------------------
# 1. One test per error category
# ---------------------------------------------------------------------------

class TestLinkerError:
    """Linker error: 'undefined reference to' and 'multiple definition of'."""

    LOG_UNDEF_REF = (
        "/usr/bin/ld: main.o: in function `main':\n"
        "main.cpp:(.text+0x1a): undefined reference to `Parser::resolveSymbol'\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    LOG_MULTI_DEF = (
        "/usr/bin/ld: symbol_table.o: multiple definition of `SymbolTable::insert';\n"
        "  first defined here\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    def test_undef_ref_error_type(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert result.error_type == "linker_error"

    def test_undef_ref_identifiers(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert "Parser::resolveSymbol" in result.identifiers

    def test_undef_ref_query_text(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert "Parser::resolveSymbol" in result.query_text()

    def test_multi_def_error_type(self):
        result = parse_log(self.LOG_MULTI_DEF)
        assert result.error_type == "linker_error"

    def test_multi_def_identifiers(self):
        result = parse_log(self.LOG_MULTI_DEF)
        assert "SymbolTable::insert" in result.identifiers


class TestCompilerError:
    """Compiler error: 'use of undeclared identifier' and 'no matching function'."""

    LOG_UNDECLARED = (
        "main.cpp:42:10: error: use of undeclared identifier 'resolveSymbol'\n"
        "    resolveSymbol(sym);\n"
        "    ^\n"
    )

    LOG_NO_MATCH = (
        "parser.cpp:88:5: error: no matching function for call to 'Parser::parse'\n"
        "    parser.parse(input, flags);\n"
        "           ^~~~~\n"
    )

    def test_undeclared_error_type(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert result.error_type == "compiler_error"

    def test_undeclared_identifiers(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert "resolveSymbol" in result.identifiers

    def test_no_match_error_type(self):
        result = parse_log(self.LOG_NO_MATCH)
        assert result.error_type == "compiler_error"

    def test_no_match_identifiers(self):
        result = parse_log(self.LOG_NO_MATCH)
        assert "Parser::parse" in result.identifiers

    def test_file_hints(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert "main.cpp" in result.file_hints


class TestIncludeError:
    """Include error: 'fatal error: X.h: No such file or directory'."""

    LOG = (
        "In file included from main.cpp:1:\n"
        "fatal error: parser/resolve.h: No such file or directory\n"
        "    #include <parser/resolve.h>\n"
        "             ^~~~~~~~~~~~~~~~~~\n"
        "compilation terminated.\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "include_error"

    def test_file_hints(self):
        result = parse_log(self.LOG)
        assert "parser/resolve.h" in result.file_hints

    def test_error_message_contains_filename(self):
        result = parse_log(self.LOG)
        assert "resolve.h" in result.error_message


class TestTemplateError:
    """Template error: 'implicit instantiation of undefined template'."""

    LOG = (
        "vector_ops.cpp:15:3: error: implicit instantiation of undefined template "
        "'std::vector<CustomType>'\n"
        "  std::vector<CustomType> v;\n"
        "  ^\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "template_error"

    def test_error_message(self):
        result = parse_log(self.LOG)
        assert "implicit instantiation" in result.error_message


class TestSegfault:
    """Segfault: 'Segmentation fault' with stack frames."""

    LOG = (
        "Program received signal SIGSEGV, Segmentation fault.\n"
        "#0  0x00005555555551a9 in Parser::resolveSymbol (this=0x0, s=...) at resolve.cpp:42\n"
        "#1  0x0000555555555230 in SymbolTable::lookup (this=0x7fffffffde10) at symbol_table.cpp:88\n"
        "#2  0x00005555555552f0 in main () at main.cpp:15\n"
    )
    LOG_NO_IN = (
        "#0 0x000055d251826584 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int)\n"
        "#1 0x000055d251826bd4 SignalHandler(int) Signals.cpp:0:0\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "segfault"

    def test_stack_frames_extracted(self):
        result = parse_log(self.LOG)
        assert len(result.stack_frames) == 3

    def test_identifiers_from_frames(self):
        result = parse_log(self.LOG)
        assert "Parser::resolveSymbol" in result.identifiers
        assert "SymbolTable::lookup" in result.identifiers

    def test_file_hints_from_frames(self):
        result = parse_log(self.LOG)
        assert "resolve.cpp" in result.file_hints
        assert "symbol_table.cpp" in result.file_hints

    def test_stack_frame_without_in_keyword(self):
        result = parse_log(self.LOG_NO_IN)
        assert result.error_type == "segfault"
        assert "llvm::sys::PrintStackTrace" in result.identifiers
        assert "SignalHandler" in result.identifiers


# ---------------------------------------------------------------------------
# 2. Multi-line log handling
# ---------------------------------------------------------------------------

class TestMultilineLog:
    """Multi-line log with noise lines surrounding the actual error."""

    LOG = (
        "make[2]: Entering directory '/home/user/project/build'\n"
        "g++ -c -o main.o main.cpp\n"
        "In file included from utils.h:3:\n"
        "main.cpp:42:10: error: use of undeclared identifier 'processData'\n"
        "    processData(buffer);\n"
        "    ^\n"
        "1 error generated.\n"
        "make[2]: *** [Makefile:15: main.o] Error 1\n"
        "make[1]: Leaving directory '/home/user/project/build'\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "compiler_error"

    def test_picks_informative_message(self):
        result = parse_log(self.LOG)
        assert "undeclared identifier" in result.error_message

    def test_identifier_extracted(self):
        result = parse_log(self.LOG)
        assert "processData" in result.identifiers


# ---------------------------------------------------------------------------
# 3. Unknown error
# ---------------------------------------------------------------------------

class TestUnknownError:
    """Log with no recognizable error pattern returns error_type='unknown'."""

    LOG = (
        "make[1]: Entering directory '/home/user/project'\n"
        "g++ -c -o main.o main.cpp\n"
        "Something went wrong but no standard pattern here\n"
    )

    def test_error_type_unknown(self):
        result = parse_log(self.LOG)
        assert result.error_type == "unknown"

    def test_identifiers_empty(self):
        result = parse_log(self.LOG)
        assert result.identifiers == []

    def test_still_returns_parsed_log(self):
        result = parse_log(self.LOG)
        assert isinstance(result, ParsedLog)
        assert result.raw_log == self.LOG


# ---------------------------------------------------------------------------
# 4. Multiple error lines — most specific wins
# ---------------------------------------------------------------------------

class TestMultipleErrorLines:
    """Log containing both include and compiler errors.

    include_error is more specific and should be returned.
    """

    LOG = (
        "main.cpp:10:5: error: use of undeclared identifier 'foo'\n"
        "    foo(bar);\n"
        "    ^\n"
        "fatal error: missing_header.h: No such file or directory\n"
        "    #include <missing_header.h>\n"
        "compilation terminated.\n"
    )

    def test_most_specific_error_wins(self):
        result = parse_log(self.LOG)
        # include_error is checked before compiler_error in priority order
        assert result.error_type == "include_error"

    def test_identifiers_from_all_errors(self):
        """Even though error_type is include_error, compiler identifiers
        should still be extracted."""
        result = parse_log(self.LOG)
        assert "foo" in result.identifiers


# ---------------------------------------------------------------------------
# 5. Deeply-namespaced identifiers (e.g. llvm::SelectionDAG::getNode)
# ---------------------------------------------------------------------------

class TestNamespacedIdentifiers:
    """Real LLVM-style log with deeply nested namespace qualifiers.

    The identifier regex must capture the full qualified name including
    all namespace levels (llvm::SelectionDAG::getNode) and must NOT
    include the mangled parameter list.
    """

    LOG = (
        "/usr/bin/ld: error: undefined reference to "
        "`llvm::SelectionDAG::getNode(unsigned int, llvm::SDLoc const&, llvm::EVT)'\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "linker_error"

    def test_full_namespace_captured(self):
        result = parse_log(self.LOG)
        assert "llvm::SelectionDAG::getNode" in result.identifiers

    def test_parameter_list_stripped(self):
        """The mangled parameter list must NOT appear in any identifier."""
        result = parse_log(self.LOG)
        for ident in result.identifiers:
            assert "(" not in ident
            assert "unsigned" not in ident


# ---------------------------------------------------------------------------
# 6. ASan error
# ---------------------------------------------------------------------------

class TestAsanError:
    """AddressSanitizer errors: heap-buffer-overflow, use-after-free, etc."""

    LOG_HEAP = (
        "=================================================================\n"
        "==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014\n"
        "READ of size 4 at 0x602000000014 thread T0\n"
        "#0 0x555555555199 in main test.cpp:10\n"
    )

    LOG_USE_AFTER_FREE = (
        "==67890==ERROR: AddressSanitizer: use-after-free on address 0x60200000001c\n"
        "WRITE of size 4 at 0x60200000001c thread T0\n"
    )

    LOG_STACK = (
        "==11111==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffd\n"
    )

    def test_heap_overflow_type(self):
        result = parse_log(self.LOG_HEAP)
        assert result.error_type == "asan_error"

    def test_use_after_free_type(self):
        result = parse_log(self.LOG_USE_AFTER_FREE)
        assert result.error_type == "asan_error"

    def test_stack_overflow_type(self):
        result = parse_log(self.LOG_STACK)
        assert result.error_type == "asan_error"

    def test_error_message_contains_asan(self):
        result = parse_log(self.LOG_HEAP)
        assert "AddressSanitizer" in result.error_message

    def test_file_hints_from_stack(self):
        result = parse_log(self.LOG_HEAP)
        assert "test.cpp" in result.file_hints


# ---------------------------------------------------------------------------
# 7. Build system error
# ---------------------------------------------------------------------------

class TestBuildSystemError:
    """CMake / Make errors."""

    LOG_CMAKE_FIND = (
        "CMake Error at CMakeLists.txt:10 (find_package):\n"
        "  Could not find package Boost\n"
    )

    LOG_MAKE_NO_RULE = (
        "make[2]: *** No rule to make target 'libfoo.a', needed by 'main'. Stop.\n"
    )

    LOG_CMAKE_GENERIC = (
        "CMake Error at /usr/share/cmake/Modules/FindPkgConfig.cmake:554:\n"
        "  Package 'glib-2.0', required by 'virtual:world', not found\n"
    )

    def test_cmake_find_type(self):
        result = parse_log(self.LOG_CMAKE_FIND)
        assert result.error_type == "build_system_error"

    def test_cmake_find_identifier(self):
        result = parse_log(self.LOG_CMAKE_FIND)
        assert "Boost" in result.identifiers

    def test_make_no_rule_type(self):
        result = parse_log(self.LOG_MAKE_NO_RULE)
        assert result.error_type == "build_system_error"

    def test_make_no_rule_identifier(self):
        result = parse_log(self.LOG_MAKE_NO_RULE)
        assert "libfoo.a" in result.identifiers

    def test_cmake_generic_type(self):
        result = parse_log(self.LOG_CMAKE_GENERIC)
        assert result.error_type == "build_system_error"

    def test_error_message_cmake(self):
        result = parse_log(self.LOG_CMAKE_FIND)
        assert "CMake Error" in result.error_message


# ---------------------------------------------------------------------------
# 8. Runtime exception
# ---------------------------------------------------------------------------

class TestRuntimeException:
    """C++ runtime exceptions: terminate, std::bad_alloc, what()."""

    LOG_TERMINATE = (
        "terminate called after throwing an instance of 'std::runtime_error'\n"
        "  what():  Connection refused\n"
        "Aborted (core dumped)\n"
    )

    LOG_BAD_ALLOC = (
        "std::bad_alloc: failed to allocate 1073741824 bytes\n"
    )

    LOG_OUT_OF_RANGE = (
        "terminate called after throwing an instance of 'std::out_of_range'\n"
        "  what():  vector::_M_range_check: __n (which is 100) >= this->size() (which is 10)\n"
    )

    def test_terminate_type(self):
        result = parse_log(self.LOG_TERMINATE)
        assert result.error_type == "runtime_exception"

    def test_terminate_identifier(self):
        result = parse_log(self.LOG_TERMINATE)
        assert "std::runtime_error" in result.identifiers

    def test_bad_alloc_type(self):
        result = parse_log(self.LOG_BAD_ALLOC)
        assert result.error_type == "runtime_exception"

    def test_bad_alloc_identifier(self):
        result = parse_log(self.LOG_BAD_ALLOC)
        assert "std::bad_alloc" in result.identifiers

    def test_out_of_range_type(self):
        result = parse_log(self.LOG_OUT_OF_RANGE)
        assert result.error_type == "runtime_exception"

    def test_error_message_contains_terminate(self):
        result = parse_log(self.LOG_TERMINATE)
        assert "terminate" in result.error_message


# ---------------------------------------------------------------------------
# 9. Expanded compiler error patterns (Fix 3)
# ---------------------------------------------------------------------------

class TestExpandedCompilerPatterns:
    """Tests for the 7 new compiler_error regex patterns."""

    LOG_NOT_MEMBER_OF = (
        "/tmp/abseil/absl/synchronization/internal/kernel_timeout.cc:143:16: "
        "error: '__BOGUS_ToTimespec' is not a member of 'absl'\n"
    )

    LOG_NO_MEMBER_NAMED = (
        "/tmp/abseil/absl/strings/cord.cc:55:10: "
        "error: 'class absl::Cord' has no member named '__BOGUS_pop_back'\n"
    )

    LOG_NOT_DECLARED_SCOPE = (
        "main.cpp:10:5: error: 'kStaticRandomData' was not declared in this scope\n"
    )

    LOG_DOES_NOT_NAME_TYPE = (
        "header.h:20:1: error: 'CordzSampleToken' does not name a type\n"
    )

    LOG_HAS_NOT_BEEN_DECLARED = (
        "test.cpp:5:10: error: 'KernelTimeout' has not been declared\n"
    )

    LOG_INCOMPLETE_TYPE = (
        "foo.cpp:12:3: error: invalid use of incomplete type 'struct Foo'\n"
    )
    LOG_NO_DECLARATION_MATCHES = (
        "foo.cpp:29:6: error: no declaration matches "
        "'bool absl::CommandLineFlag::__BOGUS_ParseFrom(std::string_view, std::string*)'\n"
    )
    LOG_FORBIDS_DECLARATION = (
        "bar.cc:35:63: error: ISO C++ forbids declaration of '__BOGUS_Utf8ForCodePoint' "
        "with no type [-fpermissive]\n"
    )

    def test_not_member_of_type(self):
        result = parse_log(self.LOG_NOT_MEMBER_OF)
        assert result.error_type == "compiler_error"

    def test_not_member_of_identifier(self):
        result = parse_log(self.LOG_NOT_MEMBER_OF)
        # __BOGUS_ should be stripped, leaving "ToTimespec"
        assert "ToTimespec" in result.identifiers

    def test_no_member_named_type(self):
        result = parse_log(self.LOG_NO_MEMBER_NAMED)
        assert result.error_type == "compiler_error"

    def test_no_member_named_identifier(self):
        result = parse_log(self.LOG_NO_MEMBER_NAMED)
        # __BOGUS_ should be stripped, leaving "pop_back"
        assert "pop_back" in result.identifiers

    def test_not_declared_scope_type(self):
        result = parse_log(self.LOG_NOT_DECLARED_SCOPE)
        assert result.error_type == "compiler_error"

    def test_not_declared_scope_identifier(self):
        result = parse_log(self.LOG_NOT_DECLARED_SCOPE)
        assert "kStaticRandomData" in result.identifiers

    def test_does_not_name_type_type(self):
        result = parse_log(self.LOG_DOES_NOT_NAME_TYPE)
        assert result.error_type == "compiler_error"

    def test_does_not_name_type_identifier(self):
        result = parse_log(self.LOG_DOES_NOT_NAME_TYPE)
        assert "CordzSampleToken" in result.identifiers

    def test_has_not_been_declared_type(self):
        result = parse_log(self.LOG_HAS_NOT_BEEN_DECLARED)
        assert result.error_type == "compiler_error"

    def test_has_not_been_declared_identifier(self):
        result = parse_log(self.LOG_HAS_NOT_BEEN_DECLARED)
        assert "KernelTimeout" in result.identifiers

    def test_incomplete_type_type(self):
        result = parse_log(self.LOG_INCOMPLETE_TYPE)
        assert result.error_type == "compiler_error"

    def test_error_message_picked_correctly(self):
        """The error message should be the line containing the pattern."""
        result = parse_log(self.LOG_NOT_MEMBER_OF)
        assert "is not a member of" in result.error_message

    def test_no_declaration_matches_identifier(self):
        result = parse_log(self.LOG_NO_DECLARATION_MATCHES)
        assert result.error_type == "compiler_error"
        assert "absl::CommandLineFlag::ParseFrom" in result.identifiers

    def test_forbids_declaration_identifier(self):
        result = parse_log(self.LOG_FORBIDS_DECLARATION)
        assert result.error_type == "compiler_error"
        assert "Utf8ForCodePoint" in result.identifiers


# ---------------------------------------------------------------------------
# 10. __BOGUS_ identifier stripping (Fix 6)
# ---------------------------------------------------------------------------

class TestBogusIdentifierStripping:
    """Synthetic errors use __BOGUS_ prefixed identifiers; those should
    be stripped to recover the real symbol name."""

    LOG_BOGUS_UNDECLARED = (
        "main.cpp:10:5: error: use of undeclared identifier '__BOGUS_StrCat'\n"
    )

    LOG_BOGUS_UNDEF_REF = (
        "/usr/bin/ld: error: undefined reference to `__BOGUS_StrAppend'\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    def test_bogus_stripped_compiler(self):
        result = parse_log(self.LOG_BOGUS_UNDECLARED)
        assert "StrCat" in result.identifiers
        assert "__BOGUS_StrCat" not in result.identifiers

    def test_bogus_stripped_linker(self):
        result = parse_log(self.LOG_BOGUS_UNDEF_REF)
        assert "StrAppend" in result.identifiers
        assert "__BOGUS_StrAppend" not in result.identifiers


class TestMojibakeQuotesNormalization:
    """Parser should extract identifiers even with mojibake quote chars."""

    LOG = (
        "/tmp/abseil/absl/log/internal/log_format.cc:174:30: "
        "error: â€˜PrefixFormatâ€™ has not been declared\n"
    )

    def test_identifier_extracted_despite_mojibake_quotes(self):
        result = parse_log(self.LOG)
        assert result.error_type == "compiler_error"
        assert "PrefixFormat" in result.identifiers


# ---------------------------------------------------------------------------
# 11. Source path extraction (Fix 4)
# ---------------------------------------------------------------------------

class TestSourcePathExtraction:
    """Tests for extract_source_paths() with and without repo_root."""

    LOG_WITH_PATH = (
        "/tmp/abseil/absl/strings/str_cat.cc:143:16: error: "
        "'__BOGUS_StrCat' is not a member of 'absl'\n"
    )

    LOG_MULTIPLE_PATHS = (
        "/tmp/abseil/absl/strings/str_cat.cc:143:16: error: foo\n"
        "/tmp/abseil/absl/strings/str_join.h:50:5: error: bar\n"
    )
    LOG_RELATIVE_PATH = (
        "absl/strings/str_cat.cc:143:16: error: foo\n"
    )
    LOG_OBJECT_PATH = (
        "absl/base/internal/low_level_alloc_test.cc.o: in function `main`\n"
    )
    LOG_CMAKE_OBJECT_PATH = (
        "cd /tmp/abseil/build/absl/flags && /usr/bin/c++ ... "
        "CMakeFiles/flags_internal.dir/internal/flag.cc.o ...\n"
    )

    LOG_NO_PATH = (
        "error: use of undeclared identifier 'foo'\n"
    )

    def test_extract_without_repo_root_fallback_to_filename(self):
        """Without repo_root, should fall back to the bare filename."""
        paths = extract_source_paths(self.LOG_WITH_PATH)
        assert "str_cat.cc" in paths

    def test_extract_with_repo_root(self):
        """With a valid repo_root whose structure matches, should normalize."""
        # Create a temp directory with a matching file structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create absl/strings/str_cat.cc within tmpdir
            target = Path(tmpdir) / "absl" / "strings" / "str_cat.cc"
            target.parent.mkdir(parents=True)
            target.touch()

            paths = extract_source_paths(self.LOG_WITH_PATH, repo_root=Path(tmpdir))
            assert "absl/strings/str_cat.cc" in paths

    def test_multiple_paths_extracted(self):
        """Should extract all unique paths from the log."""
        paths = extract_source_paths(self.LOG_MULTIPLE_PATHS)
        assert len(paths) == 2

    def test_no_paths_returns_empty(self):
        """Logs without absolute paths should return empty list."""
        paths = extract_source_paths(self.LOG_NO_PATH)
        assert paths == []

    def test_longest_suffix_preferred(self):
        """When multiple suffixes match, the longest (most specific) wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both absl/strings/str_cat.cc AND strings/str_cat.cc
            long_path = Path(tmpdir) / "absl" / "strings" / "str_cat.cc"
            long_path.parent.mkdir(parents=True)
            long_path.touch()

            paths = extract_source_paths(self.LOG_WITH_PATH, repo_root=Path(tmpdir))
            # Should prefer the longer suffix
            assert "absl/strings/str_cat.cc" in paths

    def test_relative_source_path_is_extracted(self):
        paths = extract_source_paths(self.LOG_RELATIVE_PATH)
        assert "absl/strings/str_cat.cc" in paths

    def test_object_path_is_normalized(self):
        paths = extract_source_paths(self.LOG_OBJECT_PATH)
        assert "absl/base/internal/low_level_alloc_test.cc" in paths

    def test_cmake_object_path_uses_component_context(self, tmp_path):
        target = tmp_path / "absl" / "flags" / "internal" / "flag.cc"
        target.parent.mkdir(parents=True)
        target.touch()
        paths = extract_source_paths(self.LOG_CMAKE_OBJECT_PATH, repo_root=tmp_path)
        assert "absl/flags/internal/flag.cc" in paths


class TestBuildTargetExtraction:
    LOG = (
        "/usr/bin/ld: CMakeFiles/flags_internal.dir/internal/flag.cc.o:"
        " undefined reference to `absl::CommandLineFlag::IsRetired() const'\n"
    )

    def test_extract_build_targets(self):
        targets = extract_build_targets(self.LOG)
        assert "flags_internal" in targets

    def test_parse_log_populates_build_targets(self):
        parsed = parse_log(self.LOG)
        assert "flags_internal" in parsed.build_targets


# ---------------------------------------------------------------------------
# 12. Linker: declaration outside of class (Fix 3)
# ---------------------------------------------------------------------------

class TestLinkerDeclarationOutsideClass:
    """Pattern: declaration of X outside of class is not definition."""

    LOG = (
        "foo.cpp:15:1: error: declaration of 'void Foo::bar()' "
        "outside of class is not definition\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "linker_error"

    def test_identifier_extracted(self):
        result = parse_log(self.LOG)
        assert "Foo::bar" in result.identifiers


class TestLinkerExtendedPatterns:
    """Extra linker patterns should extract useful symbols."""

    LOG_TYPEINFO = (
        "/usr/bin/ld: undefined reference to `typeinfo for absl::CommandLineFlag'\n"
    )
    LOG_INLINE_UNDEFINED = (
        "time.cc:54:40: warning: inline function "
        "'absl::{anonymous}::unix_epoch()' used but never defined\n"
    )

    def test_typeinfo_reference_extracts_symbol(self):
        result = parse_log(self.LOG_TYPEINFO)
        assert result.error_type == "linker_error"
        assert "absl::CommandLineFlag" in result.identifiers

    def test_inline_used_not_defined_classifies_linker(self):
        result = parse_log(self.LOG_INLINE_UNDEFINED)
        assert result.error_type == "linker_error"
        assert any(ident.endswith("unix_epoch") for ident in result.identifiers)
