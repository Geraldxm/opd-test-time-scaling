#!/usr/bin/env python3
"""Convert frozen math-eval raw rows into EvalPlus samples."""

import argparse
import hashlib
import json
import re
from pathlib import Path

CANONICAL = Path(__file__).resolve().parent / "data/canonical/mbpp_plus.jsonl"
CANONICAL_SHA256 = "5b07e58fcab0204278179d2062a6bcadbcb9d520298f0114f13e0c5cf3fba83e"
FIRST_CODE_BLOCK = re.compile(
    r"^[ \t]*```[^\r\n]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)


def completion_text(text: str) -> str:
    match = FIRST_CODE_BLOCK.search(text)
    if match:
        return match.group(1)
    stripped = text.lstrip()
    return stripped.removeprefix("Assistant:").lstrip() if stripped.startswith("Assistant:") else text


def problem_import_prefix(problem_text: str) -> str:
    lines = problem_text.splitlines()
    first_function = next(
        (index for index, line in enumerate(lines) if line.startswith("def ")),
        None,
    )
    return "" if first_function is None else "\n".join(lines[:first_function]) + "\n\n"


def evalplus_sample(row: dict) -> dict[str, str]:
    task_id = str(row["problem_id"])
    mode = row.get("input_mode")
    if mode == "completion":
        text = row.get("raw_text")
        if not isinstance(text, str):
            raise ValueError(f"{task_id}: completion row has no raw_text")
        return {"task_id": task_id, "completion": completion_text(text)}
    if mode == "chat":
        text = row.get("final_text")
        if not isinstance(text, str):
            raise ValueError(f"{task_id}: chat row has no final_text")
        solution = completion_text(text)
        prefix = problem_import_prefix(str(row.get("problem", "")))
        if prefix and not solution.lstrip().startswith(prefix.strip().split("\n")[0]):
            solution = prefix + solution
        return {"task_id": task_id, "solution": solution}
    raise ValueError(f"{task_id}: unsupported input_mode {mode!r}")


def raw_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.name.endswith(".inprogress"):
            raise ValueError(f"refusing unsealed raw file: {path}")
        return [path]
    root = path / "raw" if (path / "raw").is_dir() else path
    unfinished = sorted(root.rglob("*.inprogress"))
    if unfinished:
        raise ValueError(f"refusing run with in-progress raw shard: {unfinished[0]}")
    paths = sorted(root.rglob("part-*.jsonl"))
    if not paths:
        raise ValueError(f"no raw part-*.jsonl files under {root}")
    return paths


def canonical_ids() -> set[str]:
    if hashlib.sha256(CANONICAL.read_bytes()).hexdigest() != CANONICAL_SHA256:
        raise ValueError("frozen canonical problem SHA-256 mismatch")
    return {str(json.loads(line)["id"]) for line in CANONICAL.read_text(encoding="utf-8").splitlines() if line.strip()}


def convert(raw: Path, output: Path) -> int:
    rows = []
    seen: set[tuple[str, int]] = set()
    indices_by_problem: dict[str, set[int]] = {}
    for source in raw_paths(raw):
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                problem_id = str(row["problem_id"])
                sample_idx = row.get("sample_idx")
                if type(sample_idx) is not int or sample_idx < 0:
                    raise ValueError(f"{source}:{line_number}: sample_idx must be a non-negative integer")
                key = (problem_id, sample_idx)
                if key in seen:
                    raise ValueError(f"duplicate sample key {key} in {source}:{line_number}")
                seen.add(key)
                indices_by_problem.setdefault(problem_id, set()).add(sample_idx)
                rows.append(row)
    expected = canonical_ids()
    observed = set(indices_by_problem)
    if observed != expected:
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        raise ValueError(f"raw problem IDs do not match canonical data; missing={missing[:5]}, extra={extra[:5]}")
    depths = set()
    for problem_id, indices in indices_by_problem.items():
        depth = len(indices)
        if indices != set(range(depth)):
            raise ValueError(f"{problem_id}: sample_idx values must be contiguous from 0")
        depths.add(depth)
    if len(depths) != 1:
        raise ValueError(f"raw data has unequal samples per problem: {sorted(depths)}")
    rows.sort(key=lambda row: (str(row["problem_id"]), int(row["sample_idx"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(evalplus_sample(row), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = convert(args.raw, args.output)
    print(f"wrote {count} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
