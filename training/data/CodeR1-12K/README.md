# CodeR1 12K training data

This is the frozen 12,458-row CodeR1-derived training set used by the paper. The dataset comes from [ganler/code-r1-12k](https://huggingface.co/datasets/ganler/code-r1-12k), whose upstream card does not declare a license. We therefore provide a downloader and conversion code, not a copy of the dataset. The manifest pins the expected source SHA-256. The URL resolves `main`; the downloader rejects any contents that do not match the frozen hash.

## Build

Requirements: Python 3.10+, `curl`, and `sha256sum` for download.

```bash
cd training/data/CodeR1-12K
./download.sh
python -m pip install -r requirements.txt
python build.py
python build_eopd_json.py
```

This creates ignored files under `generated/`:

- `train.jsonl`: normalized records with `id`, `problem`, `entry_point`, `tests`, `reference`, `src`, and `index`.
- `eopd_train.json`: one EOPD adapter object per line, ready for the matching training input pipeline.

Both converters reject a source or output that does not match the frozen SHA-256 and expected row count in [`manifest.json`](manifest.json). The CodeR1 test split is not used. The manifest records dataset lineage and does not grant rights to the data; source data includes CodeR1 / LeetCode / TACO-derived content.
