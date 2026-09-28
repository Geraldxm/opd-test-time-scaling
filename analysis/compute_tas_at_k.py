#!/usr/bin/env python3
"""Compute the paper's TAS@K point estimate from per-question aggregates.

This standalone implementation uses Base correctness counts and the
correct-minus-incorrect token-average teacher-likelihood difference. It does
not require raw responses or model outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import TextIO


SIGNAL_FIELDS = (
    "m",  # token-average backtest and normalized DeepSeek aggregates
    "m_length_normalized_delta",  # normalized Math aggregates
    "m_avg_delta",  # Fact aggregates
    "token_average_m_delta",  # LiveCodeBench v6-only aggregate
)
OUTPUT_FIELDS = (
    "domain",
    "setting_id",
    "benchmark",
    "benchmark_id",
    "questions",
    "samples_per_question",
    "k",
    "tas_at_k",
    "tas_point_choice",
    "estimable_questions",
)


def validate_question_signal(base_n: int, base_correct: int, delta_m: float | None) -> None:
    if base_n < 1 or not 0 <= base_correct <= base_n:
        raise ValueError("invalid Base correctness counts")
    if delta_m is not None and not math.isfinite(delta_m):
        raise ValueError("token-average M delta must be finite")
    estimable = 0 < base_correct < base_n
    if estimable != (delta_m is not None):
        raise ValueError("token-average M delta must be present exactly for mixed-correctness questions")


def plugin_contribution(base_n: int, base_correct: int, delta_m: float | None, k: int) -> float:
    """Return one question's plugin finite-budget contribution to TAS@K.

    The definition is k * p * (1-p)^k * delta_m, where p is the empirical
    Base success rate and delta_m is the correct-minus-incorrect mean of the
    per-response token-average M signal. Unestimable/all-one-class questions
    contribute zero, matching the production implementation.
    """
    if isinstance(k, bool) or not isinstance(k, int) or base_n < 1 or not 1 <= k <= base_n:
        raise ValueError("k must be an integer in [1, base_n]")
    validate_question_signal(base_n, base_correct, delta_m)
    if delta_m is None or base_correct in (0, base_n):
        return 0.0
    if not math.isfinite(delta_m):
        raise ValueError("token-average M delta must be finite")
    p = base_correct / base_n
    return float(k * p * (1.0 - p) ** k * delta_m)


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("token-average M delta must be finite or empty")
    return number


def read_problem_rows(path: Path) -> tuple[list[dict[str, object]], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        available = [field for field in SIGNAL_FIELDS if field in reader.fieldnames]
        if len(available) != 1:
            raise ValueError(
                f"{path}: expected exactly one normalized-M field from {SIGNAL_FIELDS}; found {available}"
            )
        signal_field = available[0]
        required = {"base_n", "base_correct"}
        if missing := required - set(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")

        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for line, source in enumerate(reader, start=2):
            problem_id = source.get("problem_uid") or source.get("problem_id") or str(line)
            if problem_id in seen:
                raise ValueError(f"{path}:{line}: duplicate problem id {problem_id!r}")
            seen.add(problem_id)
            try:
                n = int(source["base_n"])
                correct = int(source["base_correct"])
                delta = parse_optional_float(source.get(signal_field))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: invalid per-question aggregate: {error}") from error
            if n < 1 or not 0 <= correct <= n:
                raise ValueError(f"{path}:{line}: invalid Base counts")
            try:
                validate_question_signal(n, correct, delta)
            except ValueError as error:
                raise ValueError(f"{path}:{line}: invalid per-question estimability: {error}") from error
            rows.append(
                {
                    "problem_id": problem_id,
                    "base_n": n,
                    "base_correct": correct,
                    "delta_m": delta,
                }
            )
    if not rows:
        raise ValueError(f"{path}: no per-question rows")
    depths = {int(row["base_n"]) for row in rows}
    if len(depths) != 1:
        raise ValueError(f"{path}: mixed samples-per-question values: {sorted(depths)}")
    rows.sort(key=lambda row: str(row["problem_id"]))
    return rows, signal_field


def powers_of_two(maximum: int) -> list[int]:
    result = []
    k = 1
    while k <= maximum:
        result.append(k)
        k *= 2
    return result


def calculate(
    rows: list[dict[str, object]], *, domain: str, setting_id: str,
    benchmark: str, benchmark_id: str, ks: list[int],
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("at least one per-question row is required")
    n_values = {int(row["base_n"]) for row in rows}
    if len(n_values) != 1:
        raise ValueError("all questions in a benchmark must have the same base_n")
    n = n_values.pop()
    if not ks or any(isinstance(k, bool) or not 1 <= k <= n for k in ks):
        raise ValueError(f"K values must be in [1, {n}]")
    if len(set(ks)) != len(ks):
        raise ValueError("K values must be unique")

    output = []
    for k in ks:
        values = [
            plugin_contribution(n, int(row["base_correct"]), row["delta_m"], k)
            for row in rows
        ]
        score = math.fsum(values) / len(rows)
        choice = "A" if score > 0 else "B" if score < 0 else "tie"
        output.append(
            {
                "domain": domain,
                "setting_id": setting_id,
                "benchmark": benchmark,
                "benchmark_id": benchmark_id,
                "questions": len(rows),
                "samples_per_question": n,
                "k": k,
                "tas_at_k": format(score, ".17g"),
                "tas_point_choice": choice,
                "estimable_questions": sum(row["delta_m"] is not None for row in rows),
            }
        )
    return output


def write_csv(rows: list[dict[str, object]], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def self_check() -> None:
    # Hand-calculated mixed-correctness case: 3*(2/8)*(6/8)^3*2 = 0.6328125.
    assert math.isclose(plugin_contribution(8, 2, 2.0, 3), 0.6328125, abs_tol=1e-15)
    assert plugin_contribution(8, 0, None, 3) == 0.0
    assert plugin_contribution(8, 8, None, 3) == 0.0
    for base_correct, delta in ((0, 2.0), (8, 2.0), (2, None)):
        try:
            plugin_contribution(8, base_correct, delta, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent correctness / signal state accepted")
    try:
        plugin_contribution(8, 2, 2.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid K accepted")
    # Exercise the public CSV reader too: it must reject impossible source
    # states instead of silently converting them to zero contribution.
    invalid_rows = ("q0,8,0,2.0\n", "q1,8,2,\n")
    for invalid_row in invalid_rows:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "problems.csv"
            path.write_text(
                "problem_uid,base_n,base_correct,m\n" + invalid_row,
                encoding="utf-8",
            )
            try:
                read_problem_rows(path)
            except ValueError:
                pass
            else:
                raise AssertionError("CSV reader accepted inconsistent question/signal state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, nargs="?", help="one benchmark's per-question aggregate CSV")
    parser.add_argument("--domain", help="paper domain, e.g. Math, Code, or Fact")
    parser.add_argument("--setting-id", help="frozen setting identifier")
    parser.add_argument("--benchmark", help="display name used by the paper")
    parser.add_argument("--benchmark-id", help="stable benchmark identifier")
    parser.add_argument(
        "--ks", type=int, nargs="+", help="K values (default: powers of two through base_n)"
    )
    parser.add_argument("--output", type=Path, help="output CSV (default: stdout)")
    parser.add_argument("--self-check", action="store_true", help="run dependency-free formula checks")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        print("self-check passed", file=sys.stderr)
        return 0
    if args.input_csv is None:
        parser.error("input_csv is required unless --self-check is used")
    missing = [name for name in ("domain", "setting_id", "benchmark", "benchmark_id") if not getattr(args, name)]
    if missing:
        parser.error(f"required metadata flags missing: {', '.join('--' + name.replace('_', '-') for name in missing)}")

    rows, _signal_field = read_problem_rows(args.input_csv)
    ks = args.ks or powers_of_two(int(rows[0]["base_n"]))
    result = calculate(
        rows,
        domain=args.domain,
        setting_id=args.setting_id,
        benchmark=args.benchmark,
        benchmark_id=args.benchmark_id,
        ks=ks,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            write_csv(result, handle)
    else:
        write_csv(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
