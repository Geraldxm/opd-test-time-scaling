#!/usr/bin/env python3
"""Compare compatible parsed runs using solved-set transitions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from io_utils import atomic_json, read_jsonl, stable_hash


def load_run(path: Path, k: int) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{path}: parsed run is empty")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities = {}
    decode_hashes = set()
    for row in rows:
        uid = str(row["problem_uid"])
        groups[uid].append(row)
        identities[uid] = (
            row["dataset"],
            row["gold_hash"],
            row["parser_id"],
            row["parser_config_hash"],
        )
        decode_hashes.add(
            stable_hash(
                {
                    key: value
                    for key, value in row["decode"].items()
                    if key != "samples_per_problem"
                }
            )
        )
    if len(decode_hashes) != 1:
        raise ValueError(f"{path}: run has mixed decode semantics")
    solved = set()
    for uid, problem_rows in groups.items():
        ordered = sorted(problem_rows, key=lambda row: int(row["sample_idx"]))
        if len(ordered) < k:
            raise ValueError(f"{path}: {uid} has fewer than k={k} samples")
        if any(bool(row["is_correct"]) for row in ordered[:k]):
            solved.add(uid)
    return {
        "path": str(path),
        "run_id": rows[0].get("run_id"),
        "model": rows[0]["model"],
        "identities": identities,
        "decode_hash": next(iter(decode_hashes)),
        "solved": solved,
    }


def require_compatible(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if reference["identities"] != candidate["identities"]:
        raise ValueError(
            f"incompatible problem/gold/parser semantics: {reference['path']} vs {candidate['path']}"
        )
    if reference["decode_hash"] != candidate["decode_hash"]:
        raise ValueError(
            f"incompatible decode semantics: {reference['path']} vs {candidate['path']}"
        )


def compare(base: dict[str, Any], target: dict[str, Any], teacher: dict[str, Any] | None = None) -> dict[str, Any]:
    require_compatible(base, target)
    if teacher is not None:
        require_compatible(base, teacher)
    all_problems = set(base["identities"])
    base_solved = base["solved"]
    target_solved = target["solved"]
    overlap = base_solved & target_solved
    acquisition = target_solved - base_solved
    forgotten = base_solved - target_solved
    output = {
        "schema_version": 1,
        "problem_count": len(all_problems),
        "base": {"run_id": base["run_id"], "model": base["model"], "solved": len(base_solved)},
        "target": {"run_id": target["run_id"], "model": target["model"], "solved": len(target_solved)},
        "overlap": len(overlap),
        "acquisition": len(acquisition),
        "retention": len(overlap) / len(base_solved) if base_solved else None,
        "forgetting": len(forgotten),
        "problem_uids": {
            "overlap": sorted(overlap),
            "acquisition": sorted(acquisition),
            "forgetting": sorted(forgotten),
        },
    }
    if teacher is not None:
        supported = acquisition & teacher["solved"]
        output["teacher_supported_acquisition"] = len(supported)
        output["problem_uids"]["teacher_supported_acquisition"] = sorted(supported)
    return output


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--base", type=Path, required=True)
    argument_parser.add_argument("--target", type=Path, required=True)
    argument_parser.add_argument("--teacher", type=Path)
    argument_parser.add_argument("--k", type=int, default=1)
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()
    base = load_run(args.base, args.k)
    target = load_run(args.target, args.k)
    teacher = load_run(args.teacher, args.k) if args.teacher else None
    result = compare(base, target, teacher)
    result["k"] = args.k
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
