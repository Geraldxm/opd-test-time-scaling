#!/usr/bin/env python3
"""Run one source-data chunk with the frozen LCB code-generation judge subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    with args.candidates.open(encoding="utf-8") as stream:
        candidates = json.load(stream)
    if not isinstance(candidates, list):
        raise ValueError("candidate input must be a JSON list")
    by_id = {str(row["question_id"]): row["code_list"] for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate candidate question_id")

    problems = [CodeGenerationProblem(**row) for row in source_rows]
    problems.sort(key=lambda item: str(item.question_id))
    source_ids = [str(problem.question_id) for problem in problems]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(by_id):
        raise ValueError("source/candidate question_id mismatch")
    generations = [by_id[question_id] for question_id in source_ids]
    depths = {len(items) for items in generations}
    if len(depths) != 1 or not generations or not next(iter(depths)):
        raise ValueError("every question must have the same positive sample count")

    samples = [problem.get_evaluation_sample() for problem in problems]
    # The frozen helper owns execution semantics; only verdicts are exported.
    _, results, _ = codegen_metrics(
        samples,
        generations,
        k_list=[],
        num_process_evaluate=args.workers,
        timeout=args.timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, question_id in enumerate(source_ids):
            sample_results = results[index]
            grades = [all(result > 0 for result in test_results) for test_results in sample_results]
            if len(grades) != len(generations[index]):
                raise ValueError(f"{question_id}: judge returned an incomplete sample list")
            output.write(json.dumps({"question_id": question_id, "correctness": grades}) + "\n")


if __name__ == "__main__":
    main()
