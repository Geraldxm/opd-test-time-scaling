# Paper result data

This directory contains the simplified data needed to inspect the active figures and tables in the main manuscript. It is not a raw-output or full-results archive. The CSV values preserve the precision of their frozen sources where applicable; display-rounded table values are included separately when they help match the paper.

## Contents

All row counts below exclude the CSV header.

| File | Rows | Contents |
|---|---:|---|
| `data/capacity_curves.csv` | 1,008 | Math, Code, and Fact pass@K and avg@K curves used by the main settings figures and numerical tables. Includes fractions and percentage columns. The shared Fact Base curves appear under both training-setting IDs so either Fact table block can be reconstructed directly. |
| `data/training_dynamics.csv` | 472 | All available power-of-two pass@K rows from the active Math and Code checkpoint tables: 264 Math cells and 208 Code cells. |
| `data/opd_method_variants.csv` | 264 | Main-paper Qwen3 method comparison curves. |
| `data/pass1_overview.csv` | 8 | Main-paper pass@1 and full-budget summary values; derived from `capacity_curves.csv`. Rounded columns match the table presentation. |
| `data/opd_settings.csv` | 8 | Training-setting metadata transcribed from the main-paper table. |
| `data/main_training_configurations.csv` | 3 | Main-paper training configurations. The Fact training-set field says `NQ + HotpotQA`; this names training data and is not an NQ evaluation result. |
| `data/opd_variant_hyperparameters.csv` | 5 | Hyperparameters shown in the main-paper method table. |
| `data/figure_decomposition.csv` | 220 | Grouped accuracy-bin / accuracy-change counts and pass@K contributions for the plotted Math and Fact decomposition. No question-level rows are included. |
| `data/tas_at_k_summary.csv` | 177 | Main-paper TAS@K point estimates, intervals, predicted/observed direction, and hit flags. Contains 148 direction hits. Fact rows omit NQ. MATH500 appears only as an aggregate benchmark row; no MATH500 examples or source dataset are included. |
| `data/trajectory_perplexity.csv` | 80 | Per-problem perplexity means for the trajectory-alignment figure. |
| `data/case_study_excerpt.csv` | 1 | The question and short Base/OPD excerpts shown in the text-composed case-study figure. |
| `data/coverage.csv` | 23 | Mapping for all 14 active main-paper figure labels and all 9 active table labels to their manuscript asset and included result data. Commented-out figures are excluded. |

`data/coverage.csv` records image paths relative to the manuscript's `iclr2027/` directory. The case-study figure is composed from text in the manuscript and has no image asset. `tab:pass1-overview` is represented by `pass1_overview.csv`; the numerical curve tables are represented by the corresponding rows in the curve CSVs. Configuration tables are transcribed from the active manuscript.

The training-dynamics Math rows use the frozen strict all-K checkpoint curves. Code rows are recomputed from the frozen accepted EvalPlus and LiveCodeBench per-sample verdicts, using strict `base_status == pass` and `plus_status == pass` for EvalPlus and the accepted LCB verdicts. The active manuscript's 472 numeric table cells were checked against these values; the TeX table was used as a cross-check, not as their numerical source.

## TAS@K calculator

[`../analysis/compute_tas_at_k.py`](../analysis/compute_tas_at_k.py) implements the production plugin estimator used for TAS@K. For each Base question `x`, let `p_x = base_correct / base_n` and let `delta_m_x` be the correct-minus-incorrect mean of the **per-response token-average M** signal. The question contribution and dataset score are:

```text
contribution_x(K) = K * p_x * (1 - p_x)^K * delta_m_x
TAS@K = mean_x(contribution_x(K))
```

Questions with no Base-correct samples, all Base samples correct, or an unavailable `delta_m_x` contribute zero, as in the production implementation. The mean includes every question in the benchmark.

Run the dependency-free formula check:

```bash
python analysis/compute_tas_at_k.py --self-check
```

To calculate point estimates, provide a per-question aggregate CSV for one benchmark. It must have `base_n`, `base_correct`, and exactly one token-average M delta column: `m`, `m_length_normalized_delta`, `m_avg_delta`, or `token_average_m_delta`. The column aliases match the relevant frozen aggregate schemas. These per-question inputs are deliberately not included in this release.

```bash
python analysis/compute_tas_at_k.py INPUT.csv \
  --domain Math \
  --setting-id qwen_math \
  --benchmark AIME2024 \
  --benchmark-id aime2024 \
  --ks 1 2 4 8 16 32 64 128 256 512 1024 \
  --output tas_at_k.csv
```

If `--ks` is omitted, the script uses powers of two through the input sample depth. It writes TAS@K point estimates, signs, and estimable-question counts; it does not generate confidence intervals. Recreating the published interval columns requires the omitted question-level aggregates and the source bootstrap inputs/configuration. The intervals already present in `tas_at_k_summary.csv` are the frozen paper values.

## Provenance and scope

[`release_hashes.csv`](release_hashes.csv) lists SHA-256 checksums for the 12
published figure/table CSVs and the TAS@K calculator. Paths are relative to the
public repository root. The checksums verify the released files;
[`coverage.csv`](data/coverage.csv) maps those files to main-paper figures and
tables. The underlying full experiment outputs are outside this release.

The release includes only material used by active main-paper figures and tables. It does not include a complete rollout/results bundle, raw generation outputs, the NQ evaluation benchmark/results, or the MATH500 dataset. See [`../README.md`](../README.md) for the broader release layout.
