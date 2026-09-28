# Third-party sources

The root `LICENSE` applies to code authored for this artifact. Third-party
software retains its own notices and license files; benchmark and training
data retain their source terms.

| Component | Source | Release handling |
| --- | --- | --- |
| Training `verl` fork | [Thinking-Space/Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD), based on [verl](https://github.com/verl-project/verl) | Carried byte-for-byte from review-artifact commit `c1dee277f7bd6452427a09ff5b88030f4784b17a`; its own `LICENSE` and `Notice.txt` are retained. |
| `math-eval` | [Geraldxm/math-eval](https://github.com/Geraldxm/math-eval), revision `7e4b21750fc4917649c325a092d83ba2da05f36f` | Frozen tracked source under `evaluation/math_eval/`, with its own `LICENSE` and `NOTICE`. |
| EvalPlus | [evalplus/evalplus](https://github.com/evalplus/evalplus) | Frozen judge subset copied from a G-OPD source snapshot, including a timeout safeguard in its working tree; exact file hashes and licenses are in each Code benchmark directory. |
| LiveCodeBench judge | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | Frozen subset copied from a G-OPD source snapshot; file hashes and the retained MIT license are in `evaluation/benchmarks/LiveCodeBench/`. |
| Math benchmark snapshots | [Geraldxm/math-vault](https://github.com/Geraldxm/math-vault) | Four canonical snapshots in `evaluation/benchmarks/Math/data/`; per-source terms and hashes are documented there. |
| Fact benchmark snapshots | [FlashRAG datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) | Four selected snapshots with source and selection metadata in `evaluation/benchmarks/FactQA/`. |
| CodeR1-12K training source | [ganler/code-r1-12k](https://huggingface.co/datasets/ganler/code-r1-12k) | Download/build instructions under `training/data/CodeR1-12K/`; source dataset does not declare a license. |
| Fact training source | [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) | Download/build instructions under `training/data/FactQA/`. |

For LiveCodeBench, the download script pins the upstream dataset revision and
verifies file hashes. The 4.2 GB source snapshot is not stored in Git.
