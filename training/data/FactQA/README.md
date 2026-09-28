# Fact QA training data

The paper's full closed-book Fact training pool has 169,615 examples: 79,168 from NQ and 90,447 from HotpotQA. This is training data only. The NQ evaluation dataset is not included in this public release.

The source is [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train), pinned to revision `b7d80abfee334a7a91cb377544f09180d58b34f6`. The full upstream training Parquet is about 340 MB, so the repository contains a resumable downloader and a build script rather than a copy of the data. The upstream dataset card governs its license and use.

## Build

Requirements: Python 3.10+, `curl`, and `sha256sum`.

```bash
cd training/data/FactQA
./download.sh
python -m pip install -r requirements.txt
python build.py
```

This creates ignored `generated/train.parquet`, the 169,615-row adapter used by the training pipeline. It contains one user prompt per question, uses the paper's frozen [`prompt.txt`](prompt.txt), and keeps accepted gold answers under `reward_model.ground_truth.target`. Upstream retrieval/tool prompts and gold metadata are not put into the model prompt; source metadata is retained as compact JSON in the separate `metadata` column.

The downloader verifies the upstream SHA-256 before the build. The builder checks the prompt hash, exact source row count and NQ/HotpotQA distribution, Arrow schema, and frozen output SHA-256 recorded in [`manifest.json`](manifest.json). The Parquet digest is pinned with `pyarrow==23.0.1` and zstd compression.
