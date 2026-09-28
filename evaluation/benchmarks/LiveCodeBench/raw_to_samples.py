#!/usr/bin/env python3
"""Convert sealed LCB raw rows into the judge's candidate-input JSON."""

import argparse
import json
import re
from pathlib import Path

LAST_CODE_BLOCK = re.compile(r"^[ \t]*```[^\r\n]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)", re.MULTILINE | re.DOTALL)


def code_text(raw_text: str) -> str:
    blocks = LAST_CODE_BLOCK.findall(raw_text)
    if blocks:
        return blocks[-1]
    stripped = raw_text.lstrip()
    return stripped.removeprefix("Assistant:").lstrip() if stripped.startswith("Assistant:") else raw_text


def row_text(row: dict) -> str:
    task_id = str(row["problem_id"])
    field = "raw_text" if row.get("input_mode") == "completion" else "final_text" if row.get("input_mode") == "chat" else None
    value = row.get(field) if field else None
    if not isinstance(value, str):
        raise ValueError(f"{task_id}: missing {field} for {row.get('input_mode')!r} row")
    return code_text(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=("livecodebench_v5", "livecodebench_v6_only"), required=True)
    args = parser.parse_args()
    if args.raw.is_file():
        if args.raw.name.endswith(".inprogress"):
            raise SystemExit("raw shard is still in progress; score only sealed JSONL")
        paths = [args.raw]
    else:
        if not args.raw.is_dir():
            raise SystemExit(f"raw path does not exist: {args.raw}")
        if list(args.raw.rglob("*.inprogress")):
            raise SystemExit("raw directory contains an in-progress shard; wait for sealing")
        paths = sorted(args.raw.glob("**/part-*.jsonl"))
    if not paths:
        raise SystemExit("no sealed part-*.jsonl")
    grouped: dict[str, dict[int, str]] = {}
    row_count = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row["problem_id"])
                sample_idx = row.get("sample_idx")
                if type(sample_idx) is not int or sample_idx < 0:
                    raise ValueError(f"{task_id}: sample_idx must be a nonnegative integer")
                samples = grouped.setdefault(task_id, {})
                if sample_idx in samples:
                    raise ValueError(f"{task_id}: duplicate sample_idx {sample_idx}")
                samples[sample_idx] = row_text(row)
                row_count += 1
    ordered = {}
    for task_id, samples in grouped.items():
        indices = sorted(samples)
        if indices != list(range(len(indices))):
            raise ValueError(f"{task_id}: sample_idx must be contiguous from 0, got {indices[:5]}...{indices[-5:]}")
        ordered[task_id] = [samples[index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("[\n")
        for index, task_id in enumerate(sorted(ordered)):
            if index:
                output.write(",\n")
            json.dump({"question_id": task_id, "code_list": ordered[task_id]}, output, ensure_ascii=False)
        output.write("\n]\n")
    print(f"wrote {row_count} samples for {len(ordered)} LCB questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
