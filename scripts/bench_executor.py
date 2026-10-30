"""Measure the wall time of run_python calls and write the results and the chart data.

Usage: uv run python scripts/bench_executor.py [--runs 20]

Writes results/executor-timing.json (every timing) and docs/data.json (the medians,
for the README figures and the landing page). No network access is needed.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

from data_analyst_agent.demo_script import DEMO_SCRIPT, SAMPLE_DATA_PATH
from data_analyst_agent.executor import ExecutionLimits, run_python

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "executor-timing.json"
CHART_DATA_PATH = REPO_ROOT / "docs" / "data.json"
WARMUP_RUNS = 2
LIMITS = ExecutionLimits(timeout_s=120.0, max_output_bytes=20_000)


def benchmark_cases() -> list[tuple[str, str, bool]]:
    """(label, code, uses the sample data) for every measured case."""
    demo_steps = [step for question in DEMO_SCRIPT for step in question.steps if step.code]
    labels = ["Demo: overview", "Demo: monthly chart", "Demo: growth, failing try", "Demo: growth"]
    cases = [("Empty cell", "pass", False), ("import pandas", "import pandas", False)]
    cases += [(label, step.code, True) for label, step in zip(labels, demo_steps, strict=True)]
    return cases


def time_case(code: str, with_data: bool, runs: int) -> dict:
    data_files = [SAMPLE_DATA_PATH] if with_data else []
    totals, children, exit_codes = [], [], set()
    with tempfile.TemporaryDirectory() as artifact_dir:
        for index in range(WARMUP_RUNS + runs):
            started = time.monotonic()
            result = run_python(
                code, data_files=data_files, artifact_dir=Path(artifact_dir), limits=LIMITS
            )
            total = time.monotonic() - started
            if index >= WARMUP_RUNS:
                totals.append(total)
                children.append(result.duration_s)
                exit_codes.add(result.exit_code)
    return {
        "exit_codes": sorted(exit_codes),
        "total_s": _summary(totals),
        "child_process_s": _summary(children),
        "total_runs_s": [round(value, 4) for value in totals],
    }


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "median": round(statistics.median(ordered), 4),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def chart_data(results: dict) -> dict:
    source = str(RESULTS_PATH.relative_to(REPO_ROOT))
    cases = results["cases"]
    return {
        "meta": {
            "repo": "data-analyst-agent",
            "revision": results["revision"],
            "measured": results["measured"],
        },
        "charts": [
            {
                "name": "executor-timing",
                "type": "hbar",
                "title": "Median wall time per run_python call, seconds",
                "subtitle": (
                    f"{results['runs']} runs per case after {results['warmup_runs']} warm-up "
                    f"runs, {results['machine']}"
                ),
                "source": source,
                "alt": (
                    f"An empty cell takes {cases[0]['total_s']['median']:.2f} s, importing "
                    f"pandas {cases[1]['total_s']['median']:.2f} s and the demo chart "
                    f"{cases[3]['total_s']['median']:.2f} s."
                ),
                "format": "decimal2",
                "data": [
                    {"label": case["name"], "value": round(case["total_s"]["median"], 2)}
                    for case in cases
                ],
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    cases = []
    for name, code, with_data in benchmark_cases():
        print(f"measuring {name}", file=sys.stderr)
        cases.append({"name": name, **time_case(code, with_data, args.runs)})
    results = {
        "measured": date.today().isoformat(),
        "revision": _git_revision(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs",
        "python": platform.python_version(),
        "runs": args.runs,
        "warmup_runs": WARMUP_RUNS,
        "note": "total_s is the run_python call; child_process_s is the child process alone.",
        "cases": cases,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    CHART_DATA_PATH.parent.mkdir(exist_ok=True)
    CHART_DATA_PATH.write_text(json.dumps(chart_data(results), indent=2) + "\n")
    print(
        f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and {CHART_DATA_PATH.relative_to(REPO_ROOT)}"
    )


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
