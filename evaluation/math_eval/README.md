# math-eval

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21411207.svg)](https://doi.org/10.5281/zenodo.21411207)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

Resumable, replayable math reasoning eval. Reads canonical JSONL, generates via vLLM or OpenAI-compatible API, then parses and scores independently.

```text
canonical JSONL -> evaluate.py -> raw shards -> replay_evaluation.py -> parsed + metrics
                                                        |
                                                        +-> compare_runs.py
```

Generation and scoring are decoupled: swap parsers or metrics without re-calling the model. All metrics are recomputable from raw/parsed artifacts.

## 设计选择

lm-eval-harness、LightEval 等框架在 CLI 层统一多 benchmark 调用，但每个 benchmark 各自维护数据加载、答案提取和等价判断逻辑。math-eval 将统一放在更深的两层：

| | 主流框架 | math-eval |
|---|---:|---|
| 统一层级 | CLI 参数（`--tasks`） | 数据格式（canonical JSONL）+ 验证语义（Math-Verify） |
| 添加 benchmark | 编写 task 模块 | 转换 `id/problem/answer` JSONL |
| 更换 parser | 重新调用模型 | 从 raw shards 直接解析 |
| 中断恢复 | 无 | `--resume` |

数据由 [math-vault](https://github.com/Geraldxm/math-vault) 统一维护，逐数据集记录 provenance 与许可证。

## 安装

完整 GPU 环境要求 Linux x86_64、NVIDIA driver 和 [uv](https://docs.astral.sh/uv/)。当前 lock 验证基线为 Python 3.12.11、vLLM 0.25.1+cu129 与 PyTorch 2.11.0+cu129：

```bash
./install.sh .venv
```

`requirements-gpu.lock` 是已验证环境的完整 package 快照，不是跨平台通用 lock。下载代理可通过单次命令的 `DOWNLOAD_PROXY` 环境变量显式提供。

只调用 OpenAI-compatible API 时不需要安装 vLLM、PyTorch 或 CUDA：

```bash
uv python install 3.12.11
uv venv .venv-api --python 3.12.11
uv pip install --python .venv-api/bin/python pyyaml==6.0.3 math-verify==0.9.0
```

## 数据

canonical JSONL 每行至少包含：

```json
{"id":"stable-id","problem":"...","answer":"..."}
```

仓库提供 10 条 smoke 数据：

```text
data/example_math.jsonl
```

完整数据由 [math-vault](https://github.com/Geraldxm/math-vault) 维护；`canonical/` 下的 JSONL 可直接作为本仓库输入。

## Prompt

Prompt 正文保存在 `prompts/`，YAML 只引用路径：

- completion：单个 UTF-8 文本文件，例如 `prompts/math-completion.txt`。
- chat：按 `<序号>-<role>.txt` 排序的目录，例如 `prompts/math-chat/`。

整个 prompt 必须恰好包含一次 `{{problem}}`。

## 配置

两个最短入口：

- `configs/qwen35_4b_thinking.yaml`：本地 Qwen3.5-4B thinking。
- `configs/deepseek_api.yaml`：DeepSeek API 单题。

其他公开模型示例：

```text
configs/llama32_3b_instruct.yaml
configs/llama33_70b_instruct.yaml
configs/qwen25_3b_base.yaml
configs/qwen25_3b_instruct.yaml
configs/qwen3_4b_base.yaml
configs/qwen3_4b_instruct_2507.yaml
configs/qwen3_next_80b_a3b_instruct.yaml
configs/qwen35_4b_base.yaml
```

这些配置使用公开 model ID；`model.path` 也可改为本地 checkpoint。Llama 模型仍受其上游访问条件约束。全部可配置字段见：

- `configs/reference_vllm.yaml`
- `configs/reference_openai.yaml`

常改字段为 `dataset`、`prompt`、`model`、`decode` 和 `output`。`decode.temperature`、`top_p`、`seed`、`max_output_tokens`、`samples_per_problem` 必须显式填写。每题多样本时应使用非零 temperature。

## 生成与续跑

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/qwen25_3b_instruct.yaml \
  --run-id example-vllm
```

中断后相同 config + run ID 续跑：

```bash
.venv/bin/python scripts/evaluate.py \
  --config configs/qwen25_3b_instruct.yaml \
  --run-id example-vllm --resume
```

同一数据集可由多个独立进程确定性分区。`--shard-count` 是 worker 数据分区数；配置中的 `output.shard_size` 只控制每个 raw JSONL 文件最多写多少行，两者互不影响。两张卡各运行一半数据时，保持配置中的 `tensor_parallel_size: 1`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --config configs/qwen3_4b_instruct_2507.yaml --run-id example-parallel \
  --shard-index 0 --shard-count 2 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/evaluate.py \
  --config configs/qwen3_4b_instruct_2507.yaml --run-id example-parallel \
  --shard-index 1 --shard-count 2 &
wait
```

两个进程分别写入 `example-parallel.part-00000-of-00002` 和 `example-parallel.part-00001-of-00002`。单个分片中断时，使用相同的 shard 参数并增加 `--resume`。全部完成后合并为普通 run，再沿用现有 replay：

```bash
.venv/bin/python scripts/merge_shards.py \
  --output outputs/runs/example-parallel \
  outputs/runs/example-parallel.part-00000-of-00002 \
  outputs/runs/example-parallel.part-00001-of-00002
.venv/bin/python scripts/replay_evaluation.py \
  --run-dir outputs/runs/example-parallel --k 1 10 100
```

高采样测试只需在配置中设置 `decode.samples_per_problem: 100`。跨机器运行使用相同配置、run ID、shard count 和不同 shard index，完成后将所有 part 目录复制到同一机器再显式执行合并；合并器会拒绝缺失、重复、未完成、hash 不一致、代码 revision 不一致或 Python/核心包版本不一致的分片。manifest 目前只记录 checkpoint 路径而不 fingerprint 权重内容，因此跨机器必须使用相同的 clean commit，并自行确保该路径对应完全相同的模型权重。

高采样 run 默认按每题 64 个 sample 的 band 推进；可用
`--sample-band-size 16` 调小停止粒度。vLLM 运行应只向 manifest 中的父
`process_id` 发送 SIGTERM；程序会完成当前 backend batch、封存 raw shard，
并把 manifest 标记为 `stopped`。修改配置中的
`decode.samples_per_problem` 后，以同一 run ID 加 `--resume` 可继续补齐，例如
`256 -> 512` 时只生成 `sample_idx=256..511`。

运行中可直接观察已落盘数量和所有题都达到的共同深度：

```bash
watch -n 2 'jq "{status,process_id,sample_count,target_sample_count,common_sample_depth}" outputs/runs/RUN_ID/manifests/run.json'
```

停止命令：

```bash
kill -TERM "$(jq -r .process_id outputs/runs/RUN_ID/manifests/run.json)"
```

断电、节点故障或进程被强杀时不需要先 seal。恢复后使用相同 config、run ID
和 shard 参数，加 `--resume` 即可；程序会修剪 `.jsonl.inprogress` 中损坏的
最后一行，并按 sample key 只补尚未落盘的请求。默认 `fsync_every: 1` 提供最小
断电丢失窗口。

若停止时有一个 backend batch 的少量超采样，可保留它们，并用共同前缀重放：

```bash
.venv/bin/python scripts/replay_evaluation.py \
  --run-dir outputs/runs/RUN_ID --sample-limit 256 --k 1 16 64 256
```

DeepSeek API：

```bash
.venv-api/bin/python scripts/evaluate.py \
  --config configs/deepseek_api.yaml \
  --run-id example-deepseek
```

API key 仅从 `DEEPSEEK_API_KEY` 环境变量读取，不写入配置或产物。

## 解析与判分

生成产物（raw shards）冻结后，解析与判分可独立、反复执行。更换 parser 或验证规则不需要重新调用模型。

默认使用 `math-v5.2-dual`：boxed-only strict 是正式分数，无完整 `\boxed{}` 时的全文 soft 结果只作诊断。它保持 v5.1 的抽取和判分规则，并为 prediction normalization 增加 5 秒硬超时，超时记录为 `parse_error`，避免病态 SymPy 表达式卡住整个 replay。需要复现旧合同可显式传入 `--parser-id math-v5.1-dual` 或 `--parser-id math-v5-dual`。

```bash
.venv/bin/python scripts/replay_evaluation.py \
  --run-dir outputs/runs/example-vllm --k 1 2
```

### 统一验证

主流框架为不同 benchmark 维护各自的 grader（整数比对、表达式等价、选项匹配等）。math-eval 的 canonical JSONL 将 answer 统一为字符串，因此所有 benchmark 共用同一个 Math-Verify 0.9.0 pipeline：

| 层 | 职责 |
|---|---|
| canonical data | 所有数据集 `answer` 均为字符串 |
| math-eval parser | candidate 选择、strict/soft/截断语义、状态分类、parser ID |
| Math-Verify 0.9.0 | 统一 extraction config 做数学等价判断 |
| metrics | 从冻结 verdict 重算 accuracy、pass@k、failure 率 |

gold 与 prediction 共用 `LatexExtractionConfig(boxed_match_priority=0)` 和 `ExprExtractionConfig()`。实现见 [`scripts/parser.py`](scripts/parser.py)，边界案例见 [`math_verify_walkthrough.ipynb`](math_verify_walkthrough.ipynb)。

### Strict、soft 与截断

strict 取最后一个完整 `\boxed{}` 用于正式指标；soft 仅在无完整 box 时回退全文用于诊断。truncated 与二者正交——决定 candidate 的是截断后是否仍存在完整 box：

| `final_text` 状态 | `truncated` | strict | soft |
|---|---:|---|---|
| 存在完整 `\boxed{}` | false | 验证最后一个完整 box | 同 strict |
| 存在完整 box，但之后被截断 | true | 验证最后一个完整 box | 同 strict |
| 末尾 box 未闭合，此前存在完整 box | true | 验证此前最后一个完整 box | 同 strict |
| 无完整 box，文本非空 | false/true | `no_candidate` | 全文提取验证 |
| 文本为空 | false/true | `no_candidate` | `no_candidate` |

状态字面量：`correct`、`incorrect`、`no_candidate`、`parse_error`、`verification_error`。后三者计入失败但不混为数学错误。

### 比较 run

```bash
.venv/bin/python scripts/compare_runs.py \
  --base outputs/runs/run-a/parsed/math-v5-dual/parsed.jsonl \
  --target outputs/runs/run-b/parsed/math-v5-dual/parsed.jsonl \
  --k 1 --output outputs/comparison.json
```

metrics 的 `pass@k` 使用每题全部样本的无偏估计；compare 的 solved 集合检查按 `sample_idx` 排序后的前 k 个样本。

## 产物

每个 run 位于 `outputs/runs/<run-id>/`：

- `manifests/`：config/prompt 快照、hash、环境状态。
- `raw/`：sample 级 JSONL shards；active shard 为 `.jsonl.inprogress`。
- `parsed/<parser-id>/`：逐 sample verdict。
- `metrics/<parser-id>/`：可从 parsed 重算的聚合指标。

## Notebook

- `quickstart.ipynb`：安装、生成、解析与判分主线。
- `math_verify_walkthrough.ipynb`：parser 语义、错误分类与已知边界。

两个 notebook 均不在打开时调用 GPU 或 API。

## 检查

```bash
.venv/bin/python -m unittest tests.test_evaluation_core -v
.venv/bin/python -m py_compile scripts/*.py
```

## 引用

```bibtex
@software{ge_math_eval_2026,
  author  = {Ge, Xinmu},
  title   = {math-eval: Reproducible Mathematical Reasoning Generation and Evaluation},
  year    = {2026},
  doi     = {10.5281/zenodo.21411207},
  url     = {https://github.com/Geraldxm/math-eval},
  license = {Apache-2.0}
}
```

Copyright 2026 Xinmu Ge. Licensed under [Apache License 2.0](LICENSE)。再分发须保留许可证、版权和署名声明，详见 [`NOTICE`](NOTICE)。
