# Reproduction

All commands run from the repository root after `cp .env.example .env` (see the [README](../README.md)). Every stage
skips outputs that already exist, so an interrupted script can be restarted.

## Pipeline

| Stage | Module | Device | Output (under `$CHAINUQ_ARTIFACTS`) |
|---|---|---|---|
| 0. Splits and pools | `pipeline.prepare_data` | CPU | `$CHAINUQ_DATASETS/<dataset>/hf_dataset`, `$CHAINUQ_DATASETS/pools/n<N>/` |
| 1. Generation | `pipeline.generate` | GPU (vLLM) | `gen/<backbone>/<dataset>/<split>.jsonl` (+ `.samples.jsonl` with `--n_samples 10`) |
| 2. Judge outputs | `pipeline.judge` with `scripts/serve_model.sh` | judge server | `judge/<judge>/<task>/<backbone>/<dataset>/<split>.jsonl` |
| 3. Features | `pipeline.features` | GPU | `feat/<backbone>/<dataset>/<split>/` |
| 4. Heads | `pipeline.train`, `pipeline.train_lodo` | GPU | `ckpt/<run>/`, `preds/<run>/` |
| 4b. Training-free baselines | `pipeline.cotuq`, `pipeline.scores` | GPU | `cotuq/`, `preds/<baseline>/` |
| 5. Main comparison | `pipeline.evaluate` | CPU | `results/<backbone>/eval[_test2].json`, `tables[_test2].md` |
| Analyses | `analysis.*` | CPU (GPU for `benchmark`) | `$CHAINUQ_ANALYSIS/`, `results/paper/`, `results/costs.json` |
| Paper | `reports` | CPU | `$CHAINUQ_PAPER_DIR/table/*.tex`, `fig/*.pdf` |

The unlabeled pools are generated, judged (`steps`, `final_blind`) and featurized like any `train` split: the
2,000-question pools in the main tree and the 6,000-question pools in `$CHAINUQ_POOL_ARTIFACTS`.
`python -m chainuq.pipeline.audit --backbone <b>` checks parse failures, truncation, judge outputs, label balance
and feature integrity of one backbone.

## Driver scripts

| Script | What it runs |
|---|---|
| `scripts/run_main.sh` | gpt-oss-20b server; per backbone `scripts/run_backbone.sh <b> gen\|judge\|pools\|heads`; `pipeline.evaluate` on test and test2 |
| `scripts/run_judges.sh` | inference-time judges: gpt-oss-120b (low and high reasoning effort), Qwen3-8B (thinking), Qwen3-4B correctness labels |
| `scripts/run_ablations.sh` | head ablations, UHead with PCS, distilled SAPLMA, leave-one-domain-out heads, label budgets |
| `scripts/run_thinking.sh` | thinking-mode Qwen3-8B: generation, main and trace-visible judge, features, heads |
| `scripts/run_haiku.sh`, `run_haiku_more.sh`, `run_haiku_swap_bb.sh` | Claude Haiku 5.5 as a black-box API judge (`pipeline.judge_api`, needs `ANTHROPIC_API_KEY`): imitation targets on 10k Llama source responses and the 2k pools; verdicts on validation/test |
| `scripts/run_haiku_train.sh`, `run_haiku_eval.sh` | Llama probes imitating Haiku vs gpt-oss-20b vs no imitation (`train --judge`, 10 seeds); judge-free and Haiku-at-inference evaluation |
| `scripts/run_analysis.sh` | evidence cache, Stage-2 variant runs, information decomposition, routing, fusion baselines, response-level analyses, calibrators, significance, costs and benchmark |
| `scripts/run_paper.sh` | `python -m chainuq.reports` |
| `scripts/reproduce.sh` | data preparation and all of the above, in order |

`CHAINUQ_GEN_GPU` runs generation, features and training; `CHAINUQ_JUDGE_GPU` runs the vLLM judge server (gpt-oss-120b
needs one 80 GB GPU on its own). The full reproduction takes several GPU-days on two 80 GB GPUs, mostly generation and
judging; the Stage-2 analyses run on CPU with `CHAINUQ_WORKERS` processes.

## Protocol

- Heads train with correctness labels from HotpotQA train only, early-stop on HotpotQA validation PR-AUC, and are
  reported as the mean over seeds 2026-2035 (ablations: 2026-2028; thinking mode: 2026-2030).
- Every calibrator and threshold is fit on the validation split of the evaluated dataset and applied once to its
  test split. `test2` is a fresh held-out split, evaluated once with the frozen pipeline.
- The evaluation set of a cell holds the parsed responses with a correctness label and a valid step judgment, so
  methods with and without a judge are scored on the same responses.
- Stage-2 variants are compared by cross-validation inside the validation split (`analysis.variants --cv`) before
  they are scored on the held-out splits.

## Paper outputs

`python -m chainuq.reports` writes every generated table and figure; `table/dataset.tex`, `model.tex` and
`search_space.tex` and `fig/framework.pdf` are written by hand.

All numbers quoted in the manuscript are handwritten literals at their point of use. Numeric LaTeX macros
and centralized numeric macro files are prohibited. `python -m chainuq.reports.information` prints a JSON
summary for checking and manual transcription; it does not generate numeric macros or rewrite manuscript prose.
Generated tables also contain literal numeric values.

| Output | Content | Inputs | Producers |
|---|---|---|---|
| `table/result.tex`, `app_results.tex` | main comparison (discrimination, calibration) | `results/<b>/eval.json`, `eval_extra.json` | `pipeline.evaluate`, `analysis.fusion` |
| `table/test2.tex` | Test-2 replication | `results/<b>/eval_test2.json` | `pipeline.evaluate --split test2` |
| `table/app_calibration.tex` | calibrators on the same head score | `results/paper/<b>/<d>_test.json` | `analysis.calibrators` |
| `table/significance.tex` | paired bootstrap vs. the strongest baseline and vs. fusion | `results/paper/<b>/<d>_signif.json`, `responses.json` | `analysis.significance`, `analysis.responses` |
| `table/efficiency.tex`, `efficiency_post.tex` | inference cost | `results/paper/bench_llama_hotpotqa.json`, `results/costs.json` | `analysis.benchmark`, `analysis.costs` |
| `table/selective_main.tex`, `selective.tex` | selective risk (AURC, error at 80% coverage) | `responses.json`, `predictions/` | `analysis.responses`, `analysis.predictions` |
| `table/relabel.tex` | second correctness labeler | `results/<b>/eval.json`, `responses.json` | `pipeline.evaluate`, `analysis.responses` |
| `table/decomposition.tex`, `accounting_all.tex` | information decomposition | `information/decomposition_chainuq.json` | `analysis.information` |
| `table/pools.tex`, `pools_bb.tex` | imitation targets and pool size | `runs/{attribution,target_pool_controls}.jsonl`, `information/decomposition_*_s3.json` | `analysis.variants`, `analysis.information` |
| `table/attribution.tex` | where the gains come from | `runs/attribution.jsonl` | `analysis.variants` |
| `table/ablation.tex` | Stage-2 ablations | `runs/attribution.jsonl` | `analysis.variants` |
| `table/bigjudge.tex` | judge swap | `runs/{attribution,judge_strength,cross_family_judge}.jsonl`, `information/decomposition_chainuq_*.json` | `analysis.variants`, `analysis.information` |
| `table/grid.tex` | source label budget | `runs/calibration_grid*.jsonl` | `analysis.variants` |
| `fig/teaser_{a,b,c}.pdf` | Figure 1 (three subfigures) | `information/decomposition_chainuq.json`, `information/routing_chainuq.json`, `responses.json` (fusion routing) | `analysis.information`, `analysis.routing`, `analysis.responses` |
| `fig/accepted.pdf` | errors the judge accepts | `responses.json` | `analysis.responses` |
| Standard output of `reports.information` | numerical summary for checking handwritten manuscript values | all of the above, `information/{babi_subtasks,unseen_domain}.json`, `runs/{probe_distillation,leave_one_domain_out,judge_reasoning_effort,thinking,judge_trace}.jsonl` | `analysis.subtasks`, `analysis.unseen`, `analysis.variants` |

Paths in the Inputs column are relative to `$CHAINUQ_ARTIFACTS` (`results/`) or `$CHAINUQ_ANALYSIS` (the rest).
Plots use embedded Times New Roman at 10 TeX pt and their physical paper width, so they are included without
scaling; tables contain literal numbers at the manuscript's `\small` size.
