#!/usr/bin/env python3
"""Run a one-question CPU check through conversion, isolation, judge, and metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lcb-smoke-") as temporary:
        work = Path(temporary)
        candidates = work / "candidates.json"
        output = work / "scores"
        subprocess.run(
            [sys.executable, str(ROOT / "raw_to_samples.py"), "--raw", str(ROOT / "examples/smoke/raw.jsonl"), "--output", str(candidates), "--dataset", "livecodebench_v5"],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "score_lcb.py"), "--dataset", "smoke", "--custom-outputs", str(candidates), "--output-dir", str(output), "--workers", "1", "--timeout", "3"],
            check=True,
        )
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        verdict = json.loads((output / "graded.jsonl").read_text(encoding="utf-8"))
        if metrics["question_count"] != 1 or metrics["samples_per_question"] != 1:
            raise AssertionError("smoke run did not grade one sample for one task")
        if verdict["correctness"] != [True] or metrics["pass_at_k"]["1"] != 1.0:
            raise AssertionError("known-correct sample did not pass")
    print("LiveCodeBench CPU smoke: ok")


if __name__ == "__main__":
    main()
