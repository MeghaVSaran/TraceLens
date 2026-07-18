# Evaluation Requirements

## Capability Gates

### Crash and sanitizer triage

- Parse crash, allocation, free, and thread stacks from raw reports.
- Localize the crash site and any allocator/deallocator site.
- Rank likely investigation frames without treating frame zero as the root cause.
- Preserve unknown/partial states instead of fabricating missing evidence.
- Test stripped symbols, missing source paths, multiple threads, templates, operators, and third-party frames.

### GDB and core dumps

- Collect a batch backtrace with pagination and user init files disabled.
- Record command status, timeout, tool availability, and stderr.
- Test command construction without requiring GDB in the unit suite.
- Validate end to end on Linux with a small debug-symbol fixture before claiming support.

### Preventive analysis

- Ingest findings from a real static analyzer or compiler.
- Preserve checker ID, severity, file, line, message, and trace.
- Deduplicate findings and distinguish confirmed runtime failures from possible defects.
- Measure precision on labeled vulnerable and non-vulnerable cases.

### Retrieval and RAG

- Evaluate on repository-disjoint train/dev/test splits.
- Report lexical, dense, hybrid, and reranked baselines.
- Measure root-cause file/function recall, MRR, and fix-hunk retrieval.
- Audit path leakage, duplicate issues, near-duplicate patches, and revision mismatch.

## Dataset Tiers

1. Handwritten parser fixtures for deterministic edge cases.
2. Compiled micro-programs with injected, known failures.
3. Public benchmark cases with known bug classes.
4. Real issue/log/fix-commit triples from multiple unrelated C++ repositories.

Do not use a higher metric from an easier tier to imply performance on a harder tier.

## Required Reporting

For every major evaluation, record:

- repository and exact revision;
- compiler/tool versions and flags;
- whether debug symbols and source paths were available;
- dataset size and bug-category distribution;
- excluded samples and reasons;
- accuracy, false positives, latency, and failures;
- whether any model saw related repositories or patches during training.
