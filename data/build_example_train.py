#!/usr/bin/env python3
"""Convert the one-row canonical example to Rethink-OPD's training schema."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([("index", pa.string())])),
])


def main():
    source = json.loads((HERE / "example_math.jsonl").read_text(encoding="utf-8"))
    assert all(source.get(key) for key in ("id", "problem", "answer"))
    row = {
        "data_source": "math_dapo",
        "prompt": [{"content": source["problem"] + r" Please reason step by step, and put your final answer within \boxed{{}}.", "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": source["answer"], "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": source["id"]},
    }
    output = HERE.parent / "output" / "example_math_train.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), output, compression="snappy")
    assert pq.read_table(output).to_pylist() == [row]
    print(output)


if __name__ == "__main__":
    main()
