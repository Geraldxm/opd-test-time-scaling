# Math evaluation data and examples

This directory contains the four Math benchmark snapshots used for the main
pass@K and avg@K results: AMC2023 (40 problems) and AIME2024, AIME2025,
AIME2026 (30 problems each). Each JSONL row is a byte-identical copy of the
corresponding `math-vault/canonical/` row. `data/manifest.json` records the
source path, repository revision, row count, and SHA-256. MATH500 questions are
not included in this release.

Generation, replay, parsing, and metrics use the frozen
[`math-eval`](../../math_eval/) source; its upstream repository is
[Geraldxm/math-eval](https://github.com/Geraldxm/math-eval). The frozen prompt
here matches the main manuscript and has no separate system message. The same
prompt is available for chat and completion models.

## Inference examples

Run from `evaluation/math_eval/`. The four checked configurations are
`../benchmarks/Math/configs/{amc2023,aime2024,aime2025,aime2026}_smoke.yaml`.
Each generates two responses for one problem using the public Qwen3-1.7B
model ID. Change `model.path` to a local checkpoint or model ID before use.

```bash
cd evaluation/math_eval
python scripts/evaluate.py --config ../benchmarks/Math/configs/amc2023_smoke.yaml --run-id amc2023-smoke
python scripts/replay_evaluation.py --run-dir outputs/runs/amc2023-smoke --parser-id math-v5.2-dual --k 1 2
```

Replace `amc2023` with `aime2024`, `aime2025`, or `aime2026` for each other
benchmark. To evaluate a complete dataset, remove `dataset.limit` and set
`decode.samples_per_problem: 1024`; the manuscript uses temperature 0.7,
top-p 0.95, seed 0, 32,768 context tokens, and up to 31,744 output tokens.
For completion checkpoints, set `model.input_mode: completion`, remove
`model.thinking`, and set `prompt: ../benchmarks/Math/prompts/completion.txt`.
Generation needs a GPU or a compatible inference API; replay from sealed raw
outputs is CPU runnable.

## Data attribution

The canonical files originate from [math-vault](https://github.com/Geraldxm/math-vault):

| File | Upstream data source | Upstream license as recorded by math-vault |
| --- | --- | --- |
| `amc2023.jsonl` | [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23) | Not declared |
| `aime2024.jsonl` | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | Not declared |
| `aime2025.jsonl` | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | CC BY-NC-SA 4.0 |
| `aime2026.jsonl` | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) | CC BY-NC-SA 4.0 |

The root Apache-2.0 license covers our code, not these benchmark datasets.
