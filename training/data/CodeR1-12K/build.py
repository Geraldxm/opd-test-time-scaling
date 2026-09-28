#!/usr/bin/env python3
"""Convert the frozen CodeR1-12K parquet to the normalized train.jsonl."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
SOURCE_SHA256 = MANIFEST["upstream"]["sha256"]
OUTPUT_SHA256 = MANIFEST["derived"]["train.jsonl"]["sha256"]
N_EXPECTED = MANIFEST["upstream"]["rows"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output: Path) -> None:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError(f"unexpected frozen source SHA-256: {source}")
    frame = pd.read_parquet(source)
    if len(frame) != N_EXPECTED:
        raise ValueError(f"expected {N_EXPECTED} source rows, got {len(frame)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for _, row in frame.iterrows():
                info = row["extra_info"] or {}
                task_id = row["task_id"]
                record = {
                    "id": str(task_id) if pd.notna(task_id) else str(info.get("index", "")),
                    "problem": info.get("prompt", ""),
                    "entry_point": str(row["entry_point"]) if pd.notna(row["entry_point"]) else "",
                    "tests": row["reward_model"].get("ground_truth", {}),
                    "reference": info.get("reference", ""),
                    "src": info.get("dataset", ""),
                    "index": str(info.get("index", "")),
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if sha256(temporary) != OUTPUT_SHA256:
            raise ValueError("converted train.jsonl differs from the frozen manifest")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {output}: {N_EXPECTED} rows, sha256={OUTPUT_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "downloads/train-00000-of-00001.parquet")
    parser.add_argument("--output", type=Path, default=HERE / "generated/train.jsonl")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
