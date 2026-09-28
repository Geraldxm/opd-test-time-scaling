#!/usr/bin/env python3
"""Build the frozen closed-book Fact QA EOPD training Parquet."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
SOURCE_SHA256 = MANIFEST["upstream"]["sha256"]
OUTPUT_SHA256 = MANIFEST["derived"]["train.parquet"]["sha256"]
N_EXPECTED = MANIFEST["derived"]["train.parquet"]["rows"]
PROMPT = (HERE / "prompt.txt").read_text()
CHUNK = MANIFEST["build"]["batch_rows"]

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("data_source", pa.string()),
        pa.field("prompt", pa.list_(pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())]))),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("ground_truth", pa.struct([pa.field("target", pa.list_(pa.string()))])),
                    pa.field("style", pa.string()),
                ]
            ),
        ),
        pa.field("extra_info", pa.struct([pa.field("index", pa.int64()), pa.field("split", pa.string())])),
        pa.field("metadata", pa.string()),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output: Path) -> None:
    if pa.__version__ != MANIFEST["build"]["pyarrow"]:
        raise RuntimeError(
            f"this byte-frozen adapter requires pyarrow {MANIFEST['build']['pyarrow']}; "
            f"found {pa.__version__}"
        )
    if sha256(source) != SOURCE_SHA256:
        raise ValueError(f"unexpected upstream source SHA-256: {source}")
    if hashlib.sha256(PROMPT.encode()).hexdigest() != MANIFEST["prompt"]["sha256"]:
        raise ValueError("prompt.txt differs from the frozen manifest")
    if "{{problem}}" not in PROMPT:
        raise ValueError("prompt template lacks {{problem}}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
    rows = 0
    counts = Counter()
    try:
        batch_rows = []
        for batch in pq.ParquetFile(source).iter_batches(batch_size=CHUNK):
            for index, row in enumerate(batch.to_pylist()):
                for key in ("id", "question", "golden_answers", "data_source", "ability", "reward_model", "extra_info"):
                    if key not in row:
                        raise ValueError(f"source row {rows + index} lacks {key!r}")
                if not isinstance(row["golden_answers"], list) or not all(
                    isinstance(answer, str) for answer in row["golden_answers"]
                ):
                    raise ValueError(f"source row {rows + index} has invalid golden_answers")
                reward_model = row["reward_model"]
                extra = row["extra_info"]
                if not isinstance(reward_model, dict) or not isinstance(extra, dict):
                    raise ValueError(f"source row {rows + index} has invalid adapter metadata")
                question = row["question"]
                batch_rows.append(
                    {
                        "id": row["id"],
                        "data_source": row["data_source"],
                        "prompt": [{"role": "user", "content": PROMPT.replace("{{problem}}", question)}],
                        "ability": row["ability"],
                        "reward_model": {
                            "ground_truth": {"target": row["golden_answers"]},
                            "style": reward_model["style"],
                        },
                        "extra_info": {"index": extra["index"], "split": extra["split"]},
                        "metadata": json.dumps(row.get("metadata"), ensure_ascii=False, separators=(",", ":")),
                    }
                )
                counts[row["data_source"]] += 1
                if len(batch_rows) == CHUNK:
                    writer.write_table(pa.Table.from_pylist(batch_rows, schema=SCHEMA))
                    rows += len(batch_rows)
                    batch_rows = []
        if batch_rows:
            writer.write_table(pa.Table.from_pylist(batch_rows, schema=SCHEMA))
            rows += len(batch_rows)
    finally:
        writer.close()

    try:
        expected_counts = MANIFEST["derived"]["train.parquet"]["data_source_counts"]
        if rows != N_EXPECTED or dict(sorted(counts.items())) != expected_counts:
            raise ValueError(f"unexpected rows/source distribution: rows={rows}, counts={dict(counts)}")
        if sha256(temporary) != OUTPUT_SHA256:
            raise ValueError("built train.parquet differs from the frozen manifest")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {output}: {rows} rows, sha256={OUTPUT_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "downloads/train.parquet")
    parser.add_argument("--output", type=Path, default=HERE / "generated/train.parquet")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
