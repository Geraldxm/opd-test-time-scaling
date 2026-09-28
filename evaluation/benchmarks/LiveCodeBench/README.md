# LiveCodeBench v5 and v6-only

This directory contains the paper's LiveCodeBench post-inference workflow:
frozen `math-eval` input prompts, response conversion, a frozen LiveCodeBench
code-generation judge subset, isolated execution, and pass@K/avg@K aggregation.
Inference itself is run by the frozen copy at [`../../math_eval`](../../math_eval/README.md),
whose upstream project is [math-eval](https://github.com/Geraldxm/math-eval).

## Paper benchmark scope

| Dataset | Questions | Samples per question | Frozen source files |
|---|---:|---:|---|
| LiveCodeBench v5 | 880 | 128 | `test.jsonl` through `test5.jsonl` |
| LiveCodeBench v6-only | 175 new questions | 128 | `test6.jsonl` |

The v6-only run is the new `test6.jsonl` partition, not the cumulative v1-v6
release. The upstream dataset revision and expected SHA-256 digests are recorded
in [`data/manifest.json`](data/manifest.json). The large test files are fetched
on demand from the [official Hugging Face dataset](https://huggingface.co/datasets/livecodebench/code_generation_lite)
at revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`; they are not stored in Git.
The [LiveCodeBench repository](https://github.com/LiveCodeBench/LiveCodeBench)
is linked for benchmark context; the frozen judge subset's G-OPD snapshot
provenance and per-file hashes are recorded in [`vendor/README.md`](vendor/README.md)
and [`data/manifest.json`](data/manifest.json).

## Inference

The included canonical JSONL files and prompts are the frozen generation
inputs. The example configs use the main-paper chat settings: non-thinking
Qwen3 chat, temperature 1.0, top-p 0.8, seed 0, 8,192 output tokens, and 128
samples per question. Set `model.path` to the checkpoint to evaluate.

From `evaluation/math_eval/`:

```bash
python scripts/evaluate.py \
  --config ../benchmarks/LiveCodeBench/configs/infer_v5_chat.yaml \
  --run-id lcb-v5-qwen3
python scripts/evaluate.py \
  --config ../benchmarks/LiveCodeBench/configs/infer_v6_only_chat.yaml \
  --run-id lcb-v6-only-qwen3
```

These commands require a compatible vLLM installation and model access. To
validate the config, prompt, and full canonical input without loading a model,
run this from `evaluation/math_eval/`:

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from evaluate import load_config, load_dataset
for name in ("v5", "v6_only"):
    path = Path(f"../benchmarks/LiveCodeBench/configs/infer_{name}_chat.yaml")
    config = load_config(path)
    print(name, len(load_dataset(config["dataset"])))
PY
```

## Download test data

The full v5 and v6-only judge datasets total several gigabytes. From this
directory, download and verify both releases with:

```bash
python download_data.py --dataset all
```

Download or verify one release separately:

```bash
python download_data.py --dataset v5
python download_data.py --dataset v6-only
python download_data.py --dataset all --check
```

The downloader uses only the Python standard library. It checks the pinned
SHA-256 before promoting each downloaded file into `data/source/`.

## Convert and score existing inference output

Run from this directory. `--raw` accepts one `math-eval` JSONL file or a
directory containing sealed `part-*.jsonl` shards.

```bash
python raw_to_samples.py \
  --raw /path/to/math-eval/run/raw \
  --output /tmp/lcb-v5-custom.json \
  --dataset livecodebench_v5
python score_lcb.py \
  --dataset v5 \
  --custom-outputs /tmp/lcb-v5-custom.json \
  --output-dir /path/to/results/lcb-v5
```

For v6-only, set both dataset arguments to `livecodebench_v6_only` and `v6`.
By default the scorer verifies all test-file hashes and question IDs, requires
the same positive sample count for every question, and rejects an output
directory that already exists. The paper uses 128 responses per question; the
scorer accepts other complete sample counts for small experiments.

`raw_to_samples.py` follows the paper's response extraction rule: use the last
complete fenced code block; if none exists, use the answer text after removing
only a leading `Assistant:` prefix. Chat rows use `final_text`, completion rows
use `raw_text`. Rows are sorted by question ID and sample index before being
grouped for the judge.

The scorer uses the frozen LCB code-generation test harness and
per-sample judge from [`vendor/`](vendor/README.md). It emits:

- `graded.jsonl`: one Boolean verdict per generated sample and question;
- `metrics.json`: pass@K and avg@K curves, plus dataset/judge provenance;
- `curves.csv`: the same curves in tabular form;
- `source_sha256.json`: the test-data digests used for scoring.

For each question, pass@K uses the unbiased estimator
`1 - C(n-c, K) / C(n, K)`; avg@K is the fraction correct among the first K
samples in sample-index order. The final values are macro-averages over
questions. Scoring and plotting these outputs use CPU; model generation needs
GPU resources according to the selected inference backend.

## Isolated execution requirements

Generated code is untrusted. The frozen judge runs inside this package's
Linux user, mount, network, PID, IPC, and UTS namespaces and a temporary chroot.
Each invocation uses a dedicated random host UID mapped to namespace root, and
the evaluator drops all capabilities before running candidate code. The judge
receives read-only benchmark/code inputs, no network, and CPU, address-space,
file-size, process-count, and file-descriptor limits. Scorer output promotion
is capped at 256 MiB. These RLIMITs apply per process; this package does not
provide a cgroup-wide memory or CPU cap, so lower `--workers` if aggregate
resource use needs to be reduced. The scorer fails closed if it cannot start
the isolation boundary; it never falls back to executing candidate code
directly on the host.

The per-run host UID must be able to search every ancestor directory of paths
bind-mounted into the sandbox, including the package/vendor files, benchmark
inputs, and Python environment. Keep these paths under world-searchable
ancestors or grant the UID access with an ACL; alternatively, stage the inputs
and runtime under an accessible directory. The sandbox fails closed when it
cannot traverse a required path. The final result directory is promoted by the
root-side scorer after judging.

The namespace/chroot boundary protects host resources, but candidate code and
the judge share writable `/outputs`; scoring is therefore not adversarially
tamper-proof.

Use Python 3.10–3.12 with the pinned packages in `requirements.txt`. The judge
also requires Linux, root privileges for `unshare`/`chroot`, and the `setpriv`,
`unshare`, `mount`, `chroot`, `prlimit`, and `capsh` tools. First run the
one-question CPU smoke as root in the same Python environment:

```bash
python -m pip install -r requirements.txt
python preflight.py --backend
python smoke.py
```

The smoke uses a synthetic addition question and does not download LCB data.
The vendored `reliability_guard` remains defense in depth; the Linux
isolation boundary provides the process/filesystem/network separation.
