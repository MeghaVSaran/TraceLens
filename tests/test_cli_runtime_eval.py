"""CLI tests for deterministic runtime-triage evaluation."""

import json

from click.testing import CliRunner

from src.cli.main import cli


def test_eval_runtime_prints_evidence_metrics(tmp_path):
    dataset = tmp_path / "runtime.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "overflow",
                        "log": (
                            "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
                            "#0 0x401100 in writePastEnd(int*) /repo/overflow.cpp:8:5\n"
                        ),
                        "expected": {
                            "crash_type": "heap-buffer-overflow",
                            "crash_functions": ["writePastEnd"],
                            "crash_files": ["overflow.cpp"],
                        },
                    },
                    {
                        "id": "clean",
                        "log": "program completed successfully\n",
                        "expected": {"crash_type": "unknown"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["eval-runtime", "--dataset", str(dataset)],
    )

    assert result.exit_code == 0
    assert "Runtime Triage Evaluation" in result.output
    assert "Crash type accuracy:" in result.output
    assert "1.0000" in result.output
    assert "Clean false-positive rate:" in result.output


def test_eval_runtime_json_preserves_per_case_results(tmp_path):
    dataset = tmp_path / "runtime.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "ubsan",
                    "log": "/repo/a.cpp:4:2: runtime error: signed integer overflow\n",
                    "expected": {"crash_type": "ubsan_error"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["eval-runtime", "--dataset", str(dataset), "--output", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["crash_type_accuracy"] == 1.0
    assert payload["cases"][0]["id"] == "ubsan"
