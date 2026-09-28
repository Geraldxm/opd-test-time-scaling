#!/usr/bin/env python3
"""Replay parser and metrics from durable raw rollout shards."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from io_utils import atomic_json, read_jsonl, utc_now, write_jsonl
from metrics import metrics_file
from parser import (
    DUAL_PARSER_CONFIG_HASH,
    DUAL_PARSER_ID,
    PARSER_CONFIG_HASH,
    PARSER_ID,
    V5_DUAL_PARSER_CONFIG_HASH,
    V5_DUAL_PARSER_ID,
    V51_DUAL_PARSER_CONFIG_HASH,
    V51_DUAL_PARSER_ID,
    V52_DUAL_PARSER_CONFIG_HASH,
    V52_DUAL_PARSER_ID,
    parse_and_verify,
    parse_dual_and_verify,
    parse_v5_dual_and_verify,
    parse_v51_dual_and_verify,
    parse_v52_dual_and_verify,
)

DEFAULT_PARSER_ID = V52_DUAL_PARSER_ID


def replay(
    run_dir: Path,
    k_values: list[int],
    parser_id: str = DEFAULT_PARSER_ID,
    sample_limit: int | None = None,
) -> tuple[Path, Path]:
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be >= 1")
    if parser_id not in {
        PARSER_ID,
        DUAL_PARSER_ID,
        V5_DUAL_PARSER_ID,
        V51_DUAL_PARSER_ID,
        V52_DUAL_PARSER_ID,
    }:
        raise ValueError(f"unsupported parser_id: {parser_id}")
    active_paths = sorted(run_dir.glob("raw/**/*.jsonl.inprogress"))
    if active_paths:
        raise ValueError(f"{run_dir}: unsealed raw shards remain; resume generation first")
    raw_paths = sorted(
        {
            path
            for pattern in ("raw/**/*.jsonl", "raw/**/*.jsonl.gz")
            for path in run_dir.glob(pattern)
            if not path.name.endswith(".inprogress")
        }
    )
    if not raw_paths:
        raise ValueError(f"{run_dir}: no sealed raw JSONL shards")
    manifest_path = run_dir / "manifests/run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sample_limit is not None:
        counts: dict[str, int] = {}
        for raw_path in raw_paths:
            for row in read_jsonl(raw_path):
                if int(row["sample_idx"]) < sample_limit:
                    uid = str(row["problem_uid"])
                    counts[uid] = counts.get(uid, 0) + 1
        target = manifest.get("target_samples_per_problem")
        target_count = manifest.get("target_sample_count")
        expected_problems = (
            int(target_count) // int(target)
            if target and target_count and int(target_count) % int(target) == 0
            else len(counts)
        )
        short = {uid: count for uid, count in counts.items() if count < sample_limit}
        if short or len(counts) < expected_problems:
            raise ValueError(
                f"sample_limit={sample_limit} is not available for every problem "
                f"(ready={len(counts) - len(short)}/{expected_problems}, short={short})"
            )
    parsed_path = run_dir / "parsed" / parser_id / "parsed.jsonl"
    parser_config_hash = {
        PARSER_ID: PARSER_CONFIG_HASH,
        DUAL_PARSER_ID: DUAL_PARSER_CONFIG_HASH,
        V5_DUAL_PARSER_ID: V5_DUAL_PARSER_CONFIG_HASH,
        V51_DUAL_PARSER_ID: V51_DUAL_PARSER_CONFIG_HASH,
        V52_DUAL_PARSER_ID: V52_DUAL_PARSER_CONFIG_HASH,
    }[parser_id]

    def parsed_rows():
        for raw_path in raw_paths:
            for row in read_jsonl(raw_path):
                if sample_limit is not None and int(row["sample_idx"]) >= sample_limit:
                    continue
                required = {"final_text", "gold_answer"}
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"{raw_path}: missing fields {sorted(missing)}")
                raw_row = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "answer_contract",
                        "allow_last_number",
                        "parser_id",
                        "parser_config_hash",
                        "soft",
                    }
                }
                arguments = (
                    str(row["final_text"]),
                    str(row["gold_answer"]),
                    bool(row.get("truncated", False)),
                )
                if parser_id in {
                    DUAL_PARSER_ID,
                    V5_DUAL_PARSER_ID,
                    V51_DUAL_PARSER_ID,
                    V52_DUAL_PARSER_ID,
                }:
                    parser = {
                        DUAL_PARSER_ID: parse_dual_and_verify,
                        V5_DUAL_PARSER_ID: parse_v5_dual_and_verify,
                        V51_DUAL_PARSER_ID: parse_v51_dual_and_verify,
                        V52_DUAL_PARSER_ID: parse_v52_dual_and_verify,
                    }[parser_id]
                    result = parser(*arguments)
                    yield {
                        **raw_row,
                        **asdict(result.strict),
                        "soft": asdict(result.soft),
                        "parser_id": parser_id,
                        "parser_config_hash": parser_config_hash,
                    }
                else:
                    result = parse_and_verify(*arguments)
                    yield {
                        **raw_row,
                        **asdict(result),
                        "parser_config_hash": parser_config_hash,
                    }
    count = write_jsonl(parsed_path, parsed_rows())
    metrics_path = run_dir / "metrics" / parser_id / "metrics.json"
    metrics_file([parsed_path], metrics_path, k_values)
    manifest.update(
        {
            "evaluation_status": "completed",
            "evaluated_at": utc_now(),
            "parser_id": parser_id,
            "parser_config_hash": parser_config_hash,
            "parsed_rows": count,
            "parsed_path": str(parsed_path.relative_to(run_dir)),
            "metrics_path": str(metrics_path.relative_to(run_dir)),
            "sample_limit": sample_limit,
        }
    )
    atomic_json(manifest_path, manifest)
    return parsed_path, metrics_path


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run-dir", type=Path, required=True)
    argument_parser.add_argument("--k", type=int, nargs="+", default=[1])
    argument_parser.add_argument(
        "--parser-id",
        choices=(
            PARSER_ID,
            DUAL_PARSER_ID,
            V5_DUAL_PARSER_ID,
            V51_DUAL_PARSER_ID,
            V52_DUAL_PARSER_ID,
        ),
        default=DEFAULT_PARSER_ID,
    )
    argument_parser.add_argument("--sample-limit", type=int)
    args = argument_parser.parse_args()
    parsed, metrics = replay(args.run_dir, args.k, args.parser_id, args.sample_limit)
    print(json.dumps({"parsed": str(parsed), "metrics": str(metrics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
