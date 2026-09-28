# MBPP+

Self-contained post-inference MBPP+ package: frozen benchmark data and prompt,
raw response conversion, EvalPlus scoring, and paper metrics. Inference uses
the frozen [`math-eval`](../../math_eval/README.md) engine; its upstream project
is [Geraldxm/math-eval](https://github.com/Geraldxm/math-eval).

## Contents and provenance

- `data/source/MbppPlus.jsonl`: 378 tasks from [EvalPlus MBPP+ v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0),
  SHA-256 `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`.
- `data/canonical/mbpp_plus.jsonl`: frozen inference IDs and prompts, SHA-256
  `5b07e58fcab0204278179d2062a6bcadbcb9d520298f0114f13e0c5cf3fba83e`.
- `data/canonical/mbpp_plus.metadata.json`: portable provenance, release,
  row counts, and source/canonical hashes. The scorer independently pins these
  hashes and verifies IDs before scoring or replay.
- `prompts/code-chat/` and `infer.yaml`: the paper-aligned Qwen3 chat prompt and
  an n=512 generation example.
- `raw_to_samples.py`, `score.py`, and `sandbox.py`: self-contained response
  conversion, frozen EvalPlus test execution, and Pass@K/Avg@K aggregation.
- `vendor/evalplus/`: scoring-only EvalPlus runtime with frozen file hashes and
  upstream licenses. See [`VENDORED_FROM.md`](VENDORED_FROM.md) for source and
  timeout-patch provenance.

The MBPP+ data release is covered by the included Apache-2.0 license at
`data/UPSTREAM_LICENSE`; EvalPlus scorer code includes the upstream Apache-2.0
and MIT notices. The package-level Apache-2.0 license is in `LICENSE`.

## Inference

From the public repository root, run the included n=512 example:

```bash
python evaluation/math_eval/scripts/evaluate.py \
  --config evaluation/benchmarks/MBPP+/infer.yaml \
  --run-id qwen3-1.7b-mbpp-plus-n512
```

The example uses Qwen3-1.7B chat with thinking disabled, temperature 0.7,
top-p 0.95, seed 0, a 16,384-token context, and 8,192 generated tokens. Edit
`model.path` and `model.name` for another checkpoint. vLLM generation requires
a compatible GPU; post-inference scoring below runs on CPU.

## Scoring

Install the small CPU dependencies and run the scorer from the repository root:

```bash
python -m pip install -r evaluation/benchmarks/MBPP+/requirements.txt
python evaluation/benchmarks/MBPP+/score.py \
  --raw outputs/shards/code_completion/qwen3-1.7b-mbpp-plus-n512 \
  --out-dir outputs/scored/mbpp-plus-qwen3-n512 \
  --parallel 4 \
  --wall-timeout 172800
```

The default is 4 EvalPlus workers and a 48-hour wall-clock limit for the whole
judge process. Increase `--wall-timeout` in seconds if a full n=512 run needs
longer; the limit is not restarted for each problem. The scorer accepts a raw
JSONL file, raw directory, or completed run directory. It rejects unfinished
`.inprogress` shards, missing/extra problem IDs, duplicate or non-contiguous
sample indices, unequal per-problem depths, and modified benchmark or vendor
files. It writes converted `samples.jsonl`, EvalPlus `eval_results.json`,
`summary.json`, and `summary.csv`. Choose K values with `--ks 1,4,16,128,512`;
without it, the scorer reports powers of two through the full sample depth.

To replay metrics from an existing EvalPlus result file without rerunning the
judge, use a new empty output directory:

```bash
python evaluation/benchmarks/MBPP+/score.py \
  --eval-results outputs/scored/mbpp-plus-qwen3-n512/eval_results.json \
  --out-dir outputs/scored/mbpp-plus-qwen3-n512-k16 \
  --ks 1,4,16
```

## Scoring contract

The converter takes the first complete fenced code block; if there is no
complete fence, it removes a leading `Assistant:` label and preserves the
remaining answer. Chat rows are passed to EvalPlus as `solution`, while
completion rows use `completion`. For chat responses that omit prompt imports
or header lines before the first `def`, it prepends those lines from the frozen
problem prompt.

A sample counts as correct only when both EvalPlus base and added tests pass;
`fail` and `timeout` are incorrect. Pass@K uses the unbiased estimator
`1 - C(n-c, K) / C(n, K)` per problem, averaged equally across problems. Avg@K
is the mean correctness in each problem's first K samples, also averaged
equally. The converter sorts by `sample_idx`; the frozen evaluator serializes
verdicts by `completion_id`, preserving that order for Avg@K.

## Runtime and limitations

`python evaluation/benchmarks/MBPP+/smoke_test.py` runs a tiny two-sample CPU
judge in a temporary copy; it needs no checkpoint or GPU. Full scoring needs a
root Linux launcher with `setpriv`, `unshare`, `mount`, `chroot`, `prlimit`, and
`capsh`, plus kernel support for user, mount, network, PID, IPC, and UTS
namespaces. The transient judge UID must be able to traverse every ancestor of
the mounted source, vendor, Python, and raw-input paths; private `0700`
ancestors fail closed. Use a readable checkout/input path or suitable ACLs.

The sandbox limits each process to 8 GiB address space, 900 CPU seconds, 256
processes, and 256 MiB per output file; it caps the promoted output tree at
256 MiB. It does not enforce aggregate memory or PID cgroups. Candidate code and
the EvalPlus writer share the sandbox's writable result mount while judging.
The scorer stages results separately, rejects extra files, and prevents that
mount from overwriting trusted samples or summaries, but this does not prove
verdicts cannot be maliciously tampered with inside the judge process.

This package freezes a reproducible current judge with documented source and
patch hashes. Historical accepted run manifests do not record the EvalPlus
working-tree source hash, so this package cannot establish byte-for-byte
identity with every historical scoring run; see `VENDORED_FROM.md`.
