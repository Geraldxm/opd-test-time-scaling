# FactQA

Self-contained scoring package for the four FactQA datasets reported in the main paper:
HotpotQA, TriviaQA, PopQA, and 2WikiMultiHopQA. Each frozen evaluation set has
200 questions and gold-answer aliases. This package contains no NQ evaluation
set and no smoke set.

## Files and provenance

- `data/canonical/`: the four canonical JSONL datasets used by inference and scoring.
- `prompts/chat/00-user.txt`: frozen user prompt; only the question is sent to the model.
- `score.py`: standalone `fact-v3` answer extraction, exact-match scoring, and Pass@K / Avg@K aggregation.
- `data/selection_provenance.json`: pinned upstream revision, source file hashes, selection rules, and selected source IDs / question hashes.
- `data_manifest.json`: file hashes and the scorer/prompt versions.

The selected questions came from [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets), pinned at revision [`bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c). The upstream dataset declares **CC-BY-SA-4.0**. Split files: `hotpotqa/dev.jsonl`, `triviaqa/test.jsonl`, `popqa/test.jsonl`, and `2wikimultihopqa/dev.jsonl`. Our frozen selection, gold aliases, and source hashes are recorded in `data/selection_provenance.json`.

The prompt asks for step-by-step reasoning and one final `<answer>...</answer>` span. Gold aliases are present only as scoring metadata and are not sent to inference.

## Inference

Generation uses the included frozen [`math-eval`](../../math_eval/README.md) code. Its upstream project and documentation are at [Geraldxm/math-eval](https://github.com/Geraldxm/math-eval). From the public repository root, run any of the four provided configs (one config per main-paper dataset):

| Dataset | Inference config | Run ID |
|---|---|---|
| HotpotQA | `evaluation/benchmarks/FactQA/examples/infer_hotpotqa.yaml` | `qwen3-1.7b-hotpotqa-n1024` |
| TriviaQA | `evaluation/benchmarks/FactQA/examples/infer_triviaqa.yaml` | `qwen3-1.7b-triviaqa-n1024` |
| PopQA | `evaluation/benchmarks/FactQA/examples/infer_popqa.yaml` | `qwen3-1.7b-popqa-n1024` |
| 2WikiMultiHopQA | `evaluation/benchmarks/FactQA/examples/infer_2wikimultihopqa.yaml` | `qwen3-1.7b-2wikimultihopqa-n1024` |

For example, run the PopQA config with:

```bash
python evaluation/math_eval/scripts/evaluate.py \
  --config evaluation/benchmarks/FactQA/examples/infer_popqa.yaml \
  --run-id qwen3-1.7b-popqa-n1024
```

The configs use the public `Qwen/Qwen3-1.7B` checkpoint, chat mode with thinking disabled, temperature 0.7, top-p 0.95, and 1,024 responses per question. Use a vLLM-compatible GPU for local checkpoint inference. An OpenAI-compatible inference endpoint can be used instead by changing the model backend/configuration; that path does not require a local GPU. The full evaluation produces 204,800 responses per dataset, so generation requires substantial compute and storage.

For example, score the completed PopQA run directly from the public repository root:

```bash
python evaluation/benchmarks/FactQA/score.py \
  --run-dir outputs/shards/qwen3-1.7b-popqa-n1024 \
  --dataset evaluation/benchmarks/FactQA/data/canonical/popqa.jsonl \
  --samples-per-problem 1024
```

If inference was launched with multiple math-eval partitions, merge every completed partition first using `evaluation/math_eval/scripts/merge_shards.py`. The scorer rejects incomplete or unmerged runs, active `.inprogress` files, dataset/hash mismatches, duplicate/missing sample indices, and problem-text mismatches. Scoring and aggregation need only Python's standard library and CPU; no model weights or GPU are needed after raw outputs exist.

The scorer writes `parsed/fact-v3/parsed.jsonl`, `metrics.json`, and `manifest.json` under the run directory. Outputs record relative raw-file names and SHA-256 hashes, so they do not embed machine-specific workspace paths.

## Scoring contract

`fact-v3` accepts exactly one correctly ordered, nonempty `<answer>...</answer>` pair in the generated text. It allows other text before or after the pair; nested angle brackets inside the answer, missing/repeated tags, and empty answers fail extraction. It normalizes the whole extracted answer and each gold alias by lowercasing, removing ASCII punctuation, removing English articles (`a`, `an`, `the`), and collapsing whitespace. Exact equality with any alias is required; there is no substring matching, semantic judging, candidate splitting, or fallback when tags are invalid. Invalid extractions remain failures in the full sample denominator.

For each K, **Pass@K** uses the unbiased estimator `1 - C(n-c, K) / C(n, K)`, where `n` is the number of responses for a question and `c` is its correct count. **Avg@K** is the mean accuracy among the first K samples, ordered by `sample_idx`. Both metrics average equally over questions. By default, the scorer reports K values 1, 2, 4, ..., up to `n` (and includes `n` if it is not a power of two); pass a custom list with `--k 1 8 32 128`.

The scorer also reports extraction rate, truncation rate, generated `<think>` tag count, and final-line format validity as diagnostics. Truncation and final-line format do not independently change exact-match scoring.

## Checks

```bash
python evaluation/benchmarks/FactQA/score.py --check-package
python evaluation/benchmarks/FactQA/score.py --self-check
```

The CPU self-check creates a temporary two-sample run and verifies answer extraction, exact-match, Pass@K, and Avg@K without a model or GPU.
