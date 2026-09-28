# HumanEval+

Self-contained post-inference HumanEval+ package: frozen data and prompt, raw
response conversion, EvalPlus scoring, and paper metrics. Inference uses the
frozen [`math-eval`](../../math_eval/README.md) engine; its upstream project is
[Geraldxm/math-eval](https://github.com/Geraldxm/math-eval).

## Contents and provenance

- `data/source/HumanEvalPlus.jsonl`: 164 tasks from [EvalPlus HumanEval+
  v0.1.10](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10),
  SHA-256 `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- `data/canonical/humaneval_plus.jsonl`: frozen inference problem IDs and
  prompts, SHA-256
  `7fee5dcd0f1f438df47d4eab083c5b0d4ba4c68881e795f7455a65eb2a80b71b`.
- `data/canonical/humaneval_plus.metadata.json`: portable source paths, release,
  row counts, and both data hashes. The scorer pins the hashes independently
  and checks source IDs against canonical IDs before scoring or replay.
- `prompts/code-chat/` and `infer.yaml`: paper-aligned chat prompt and example
  generation settings.
- `raw_to_samples.py`, `score.py`, and `sandbox.py`: self-contained conversion,
  frozen EvalPlus test execution, and pass@K/avg@K aggregation.
- `vendor/evalplus/`: the scoring-only EvalPlus runtime subset with file hashes
  and upstream licenses. Provenance and the known timeout patch are in
  [`VENDORED_FROM.md`](VENDORED_FROM.md).

The data is released by EvalPlus under Apache-2.0; the HumanEval+ release also
states that it complies with MIT because it builds on OpenAI HumanEval. License
copies and notices are included in `data/UPSTREAM_LICENSE` and
`vendor/evalplus/LICENSE-MIT`. EvalPlus scoring code is covered by the included
Apache-2.0 and MIT notices.

## Inference

From the public repository root, run the included n=512 example:

```bash
python evaluation/math_eval/scripts/evaluate.py \
  --config evaluation/benchmarks/HumanEval+/infer.yaml \
  --run-id qwen3-1.7b-humaneval-plus-n512
```

The example uses Qwen3-1.7B chat with thinking disabled, temperature 0.7,
top-p 0.95, seed 0, a 16,384-token context, and 8,192 generated tokens. Edit
`model.path` and `model.name` for another checkpoint. Local vLLM generation needs
a compatible GPU; the post-inference judge below runs on CPU.

## Scoring

Install the small CPU dependencies and run the scorer from the repository root:

```bash
python -m pip install -r evaluation/benchmarks/HumanEval+/requirements.txt
python evaluation/benchmarks/HumanEval+/score.py \
  --raw outputs/shards/code_completion/qwen3-1.7b-humaneval-plus-n512 \
  --out-dir outputs/scored/humaneval-plus-qwen3-n512 \
  --parallel 4 \
  --wall-timeout 172800
```

The default is 4 EvalPlus workers and a 48-hour wall-clock limit for the whole
judge process. Increase `--wall-timeout` in seconds if a full n=512 run needs
longer; the limit is not restarted for each problem.

The scorer accepts a raw JSONL file, raw directory, or completed run directory.
It rejects unfinished `.inprogress` shards, missing/extra problem IDs, duplicate
or non-contiguous sample indices, unequal per-problem depths, and modified data
or vendor files. It writes converted `samples.jsonl`, frozen EvalPlus
`eval_results.json`, `summary.json`, and `summary.csv`. Choose other K values with
`--ks 1,4,16,128,512`; without it, the scorer reports powers of two and includes
the full sample depth even when it is not a power of two.

To recalculate metrics without rerunning the judge, pass a previous EvalPlus
result file and use a new empty output directory:

```bash
python evaluation/benchmarks/HumanEval+/score.py \
  --eval-results outputs/scored/humaneval-plus-qwen3-n512/eval_results.json \
  --out-dir outputs/scored/humaneval-plus-qwen3-n512-k16 \
  --ks 1,4,16
```

## Scoring contract

The converter matches the frozen paper converter (SHA-256
`b53fc3ff0cc8941db4e10d9f524c5e9262d0369bfd237f443e89949f961768d0`): it takes
the first complete fenced block; when no complete fence exists, it removes a
leading `Assistant:` label and preserves the remaining text. Chat rows are sent
to EvalPlus as `solution`; completion rows are sent as `completion`. For chat
responses, if the generated code omits the prompt's import/header lines before
the first `def`, the converter prepends those lines so the result has the same
scaffolding as prompt-plus-completion evaluation.

Each sample must pass **both** EvalPlus base and added tests to count as correct;
`fail` and `timeout` are incorrect. Pass@K uses the unbiased estimator
`1 - C(n-c, K) / C(n, K)` per problem, averaged equally over problems. Avg@K is
the mean correctness among each problem's first K samples, also averaged equally
over problems. The converter sorts by `sample_idx`; the frozen evaluator stores
verdicts sorted by `completion_id`, so `eval_results.json` preserves that order
for Avg@K replay.

## Runtime and limitations

`python evaluation/benchmarks/HumanEval+/smoke_test.py` runs a tiny two-sample
CPU judge in a temporary copy; it needs no checkpoint or GPU. Full scoring
requires a root Linux launcher with `setpriv`, `unshare`, `mount`, `chroot`,
`prlimit`, and `capsh`, and kernel support
for user, mount, network, PID, IPC, and UTS namespaces. The transient judge UID
must be able to traverse every ancestor of the mounted source, vendor, Python,
and raw-input paths; private `0700` ancestors will fail closed. Use a readable
checkout/input path or suitable ACLs.

The sandbox limits each process to 8 GiB address space, 900 CPU seconds, 256
processes, and 256 MiB per output file, and caps the promoted output tree at
256 MiB. It does not enforce aggregate memory or PID cgroups. Candidate code and
the EvalPlus writer share the sandbox's writable result mount during judging;
the scorer stages results separately, rejects extra files, and never lets that
mount overwrite `samples.jsonl` or summaries, but this is not a proof against
malicious verdict tampering inside the judge process.

The package freezes a reproducible current judge with documented source and
patch hashes. Historical accepted run manifests do not record the EvalPlus
working-tree source hash, so this package does not claim byte-for-byte identity
with every historical scoring run; see `VENDORED_FROM.md`.
