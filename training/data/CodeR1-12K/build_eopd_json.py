#!/usr/bin/env python3
"""Convert normalized CodeR1-12K JSONL to the EOPD training adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
SOURCE_SHA256 = MANIFEST["derived"]["train.jsonl"]["sha256"]
OUTPUT_SHA256 = MANIFEST["derived"]["eopd_train.json"]["sha256"]
N_EXPECTED = MANIFEST["upstream"]["rows"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output: Path) -> None:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError(f"unexpected normalized source SHA-256: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    count = 0
    try:
        with source.open(encoding="utf-8") as input_handle, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                row = json.loads(line)
                record = {
                    "data_source": "code_r1_12k",
                    "prompt": [{"role": "user", "content": row["problem"]}],
                    "ability": "CODE",
                    "reward_model": {"style": "rule", "ground_truth": row["tests"]},
                    "extra_info": {
                        "index": row["id"],
                        "dataset": row["src"],
                        "reference": row["reference"],
                    },
                }
                output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        if count != N_EXPECTED:
            raise ValueError(f"expected {N_EXPECTED} rows, got {count}")
        if sha256(temporary) != OUTPUT_SHA256:
            raise ValueError("converted eopd_train.json differs from the frozen manifest")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {output}: {count} rows, sha256={OUTPUT_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "generated/train.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "generated/eopd_train.json")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
