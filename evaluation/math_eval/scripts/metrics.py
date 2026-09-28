#!/usr/bin/env python3
"""Recompute single-run math evaluation metrics from parsed JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from io_utils import atomic_json, read_jsonl


REPORT_STATUSES = (
    "correct",
    "incorrect",
    "no_candidate",
    "parse_error",
    "verification_error",
)
PARSER_FAILURE_STATUSES = frozenset(
    {"no_candidate", "parse_error", "verification_error"}
)
PARSER_ERROR_STATUSES = frozenset({"parse_error", "verification_error"})
VERDICT_FIELDS = (
    "status",
    "is_correct",
    "candidate_text",
    "normalized_prediction",
    "normalized_gold",
    "extraction_rule",
    "error",
)


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    if correct_count == 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct_count, k) / math.comb(sample_count, k)


def _score_metrics(
    materialized: list[dict[str, Any]],
    k_values: list[int],
    status_value,
    correct_value,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        groups[str(row["problem_uid"])].append(row)

    per_k = {}
    for k in k_values:
        eligible = []
        averages = []
        for problem_rows in groups.values():
            ordered = sorted(problem_rows, key=lambda row: int(row["sample_idx"]))
            if len(ordered) < k:
                continue
            correct = sum(correct_value(row) for row in ordered)
            eligible.append(pass_at_k(len(ordered), correct, k))
            averages.append(mean(correct_value(row) for row in ordered[:k]))
        if eligible:
            per_k[str(k)] = {
                "pass_at_k": mean(eligible),
                "avg_at_k": mean(averages),
                "problem_count": len(eligible),
            }

    statuses = Counter(status_value(row) for row in materialized)
    status_keys = sorted(set(REPORT_STATUSES) | set(statuses))
    evaluation_failure_count = sum(
        statuses[status] for status in PARSER_FAILURE_STATUSES
    )
    parser_error_count = sum(statuses[status] for status in PARSER_ERROR_STATUSES)
    return {
        "accuracy": mean(correct_value(row) for row in materialized),
        "status_counts": {key: statuses[key] for key in status_keys},
        "status_rates": {
            key: statuses[key] / len(materialized) for key in status_keys
        },
        "evaluation_failure_count": evaluation_failure_count,
        "evaluation_failure_rate": evaluation_failure_count / len(materialized),
        "parser_error_count": parser_error_count,
        "parser_error_rate": parser_error_count / len(materialized),
        "k": per_k,
    }


def _validate_rows(
    materialized: list[dict[str, Any]], k_values: list[int]
) -> tuple[list[int], bool]:
    if not materialized:
        raise ValueError("parsed input is empty")
    if not k_values or any(isinstance(k, bool) or int(k) < 1 for k in k_values):
        raise ValueError("k values must be positive integers")
    normalized_k = sorted({int(k) for k in k_values})

    required = {
        "dataset",
        "model",
        "problem_uid",
        "sample_idx",
        "status",
        "is_correct",
        "parser_id",
    }
    seen = set()
    for index, row in enumerate(materialized, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"parsed row {index} is missing fields: {sorted(missing)}")
        key = (str(row["problem_uid"]), int(row["sample_idx"]))
        if key in seen:
            raise ValueError(f"duplicate parsed sample key: {key}")
        seen.add(key)
        status = str(row["status"])
        if status not in REPORT_STATUSES or bool(row["is_correct"]) != (
            status == "correct"
        ):
            raise ValueError(f"invalid strict verdict at {key}")

    for field in ("run_id", "dataset", "model", "parser_id", "parser_config_hash"):
        if len({row.get(field) for row in materialized}) != 1:
            raise ValueError(f"parsed input has mixed {field}")

    soft_flags = [isinstance(row.get("soft"), dict) for row in materialized]
    if any(soft_flags) and not all(soft_flags):
        raise ValueError("parsed input mixes single and dual verdict rows")
    is_dual = all(soft_flags)
    if is_dual:
        for row in materialized:
            key = (str(row["problem_uid"]), int(row["sample_idx"]))
            soft = row["soft"]
            missing = set(VERDICT_FIELDS) - soft.keys()
            if missing:
                raise ValueError(f"soft verdict at {key} is missing: {sorted(missing)}")
            soft_status = str(soft["status"])
            if soft_status not in REPORT_STATUSES or bool(soft["is_correct"]) != (
                soft_status == "correct"
            ):
                raise ValueError(f"invalid soft verdict at {key}")
            if bool(row["is_correct"]) and not bool(soft["is_correct"]):
                raise ValueError(f"strict correct but soft incorrect at {key}")
            if row["status"] != "no_candidate" and any(
                row.get(field) != soft.get(field) for field in VERDICT_FIELDS
            ):
                raise ValueError(f"strict and soft candidate verdicts differ at {key}")
    return normalized_k, is_dual


def compute_metrics(rows: Iterable[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    materialized = list(rows)
    k_values, is_dual = _validate_rows(materialized, k_values)
    output_lengths = [
        int(row["output_tokens"])
        for row in materialized
        if row.get("output_tokens") is not None
    ]
    common = {
        "run_id": materialized[0].get("run_id"),
        "dataset": materialized[0]["dataset"],
        "model": materialized[0]["model"],
        "parser_id": materialized[0]["parser_id"],
        "parser_config_hash": materialized[0].get("parser_config_hash"),
        "problem_count": len({str(row["problem_uid"]) for row in materialized}),
        "sample_count": len(materialized),
        "truncation_rate": mean(bool(row.get("truncated")) for row in materialized),
        "mean_output_tokens": mean(output_lengths) if output_lengths else None,
    }
    strict = _score_metrics(
        materialized,
        k_values,
        lambda row: str(row["status"]),
        lambda row: bool(row["is_correct"]),
    )
    if not is_dual:
        return {
            "schema_version": 2,
            **common,
            "status_counts": strict["status_counts"],
            "status_rates": strict["status_rates"],
            "parser_failure_count": strict["evaluation_failure_count"],
            "parser_failure_rate": strict["evaluation_failure_rate"],
            "k": strict["k"],
        }

    soft = _score_metrics(
        materialized,
        k_values,
        lambda row: str(row["soft"]["status"]),
        lambda row: bool(row["soft"]["is_correct"]),
    )
    complete_box_count = sum(
        row.get("extraction_rule") == "last_boxed" for row in materialized
    )
    no_candidate_counts = {"empty_final_text": 0, "no_complete_box": 0}
    for row in materialized:
        if row["status"] == "no_candidate":
            reason = (
                "empty_final_text"
                if not str(row.get("final_text", "")).strip()
                else "no_complete_box"
            )
            no_candidate_counts[reason] += 1
    soft_recovery_count = sum(
        not bool(row["is_correct"]) and bool(row["soft"]["is_correct"])
        for row in materialized
    )
    interactions = Counter(
        (
            str(row["status"]),
            str(row["soft"]["status"]),
            bool(row.get("truncated")),
        )
        for row in materialized
    )
    return {
        "schema_version": 3,
        **common,
        "strict": strict,
        "soft": soft,
        "complete_box_count": complete_box_count,
        "complete_box_rate": complete_box_count / len(materialized),
        "no_candidate_counts": no_candidate_counts,
        "soft_recovery_count": soft_recovery_count,
        "soft_recovery_rate": soft_recovery_count / len(materialized),
        "interaction_counts": [
            {
                "strict_status": strict_status,
                "soft_status": soft_status,
                "truncated": truncated,
                "count": count,
            }
            for (strict_status, soft_status, truncated), count in sorted(
                interactions.items()
            )
        ],
    }


def metrics_file(inputs: list[Path], output: Path, k_values: list[int]) -> dict[str, Any]:
    rows = [row for path in inputs for row in read_jsonl(path)]
    metrics = compute_metrics(rows, k_values)
    atomic_json(output, metrics)
    return metrics


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", type=Path, nargs="+", required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--k", type=int, nargs="+", default=[1])
    args = argument_parser.parse_args()
    metrics = metrics_file(args.input, args.output, args.k)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
