# Artifacts

Every stage reads and writes under `$CHAINUQ_ARTIFACTS` through the path functions of
[`chainuq/artifacts.py`](../chainuq/artifacts.py). Splits are `train`, `validation`, `test` and `test2`; the train
splits of the OOD datasets are the unlabeled pools.

```text
gen/<backbone>/<dataset>/<split>.jsonl                      greedy generations (+ .stats.json)
gen/<backbone>/<dataset>/<split>.samples.jsonl              10 temperature samples (Semantic Entropy, SelfCheckGPT)
judge/<judge>/<task>/<backbone>/<dataset>/<split>.jsonl     judge outputs (+ .stats.json)
feat/<backbone>/<dataset>/<split>/shard_*.pt                cached features (+ DONE.json with backbone dimensions)
cotuq/<backbone>/<dataset>/<split>.jsonl                    CoT-UQ keywords
ckpt/<run>/<backbone>/seed<k>/model.pt, meta.json           trained heads and their training summary
preds/<run>/<backbone>/seed<k>/<dataset>/<split>.jsonl      per-response scores of every method
results/                                                    evaluation outputs (below)
```

`$CHAINUQ_POOL_ARTIFACTS` has the same layout for the 6,000-question pools (`gen/`, `judge/`, `feat/` of the `train`
splits of MuSiQue, bAbI, GSM8K and MATH).

## Record formats

**Generation** (`gen/`): `uid`, `question`, `user_content`, `gold`, `gold_aliases`, `n_hops`, `prompt_ids`, `gen_ids`,
`gen_text`, `finish_reason`, `pred_answer`, `label_em` (answer-matching datasets), `parse_ok`, `drop_reason`, and, if
parsed, `steps` / `conclusion` as `{"text", "tok": [start, end)}` in generated-token coordinates.

**Judge** (`judge/`): `uid`, `value` (0/1 for `correct` and `final_blind`, a 0/1 list per step for `steps`, `null` if
the output did not parse), `raw`, token counts; `final_blind` also has `p_verbal` (verbalized probability) and
`p_true` (P(true) from token log-probabilities). Judge keys are defined in
[`registry.py`](../chainuq/registry.py): `gptoss` (main), `gptoss-trace`, `gptoss120b`, `gptoss120b-high`,
`qwen3judge` and `qwen4b`.

**Features** (`feat/`): a dict `uid -> tensors`; see the docstring of
[`pipeline/features.py`](../chainuq/pipeline/features.py).

**Predictions** (`preds/`): `uid`, `logit` (raw confidence score; higher = more likely correct), `step_logits` (one
score per reasoning step, or `null`); PCS heads add `verdict_hat` (log-odds of an accepting verdict level) and,
for the ChainUQ head, `verdict_logq` (log-probabilities of the four verdict levels).

## Runs

A run is a head key of [`heads/__init__.py`](../chainuq/heads/__init__.py), with the suffix of
`pipeline.train --tag` if any.

| Run | Meaning |
|---|---|
| `chainuq` | ChainUQ head: PCS on labeled HotpotQA and the 6,000-question pools |
| `chainuq-source` | PCS on labeled HotpotQA only |
| `chainuq-pool2k` | PCS with the 2,000-question pools |
| `chainuq-pool-steps`, `chainuq-pool-verdict` | only the step labels / only the verdict as pool targets |
| `chainuq-wo-pcs` | correctness labels only |
| `chainuq-lodo-<d>` | pools of every OOD domain except `<d>` (`pipeline.train_lodo`) |
| `<run>-lf01`, `-lf05`, `-lf25` | correctness labels on 1% / 5% / 25% of HotpotQA train (`--label_frac`) |
| `uhead-pcs`, `uhead-pcs-pool6k` | UHead with the PCS targets, without and with the 6,000-question pools |
| `saplma-pool6k-pseudo` | SAPLMA trained on the pool verdicts as soft correctness labels |
| `saplma`, `factoscope`, `lookback`, `uhead`, `meanpool-mlp`, `attnpool-mlp` | supervised probe baselines |
| `mcp`, `ppl`, `mte`, `ccp`, `cotuq`, `se`, `selfcheck` | training-free baselines |
| `judge-step`, `judge-final`, `rm` | judge and reward-model baselines |

## Results

```text
results/<backbone>/eval.json, eval_test2.json   main comparison: datasets.<d>.{tier_a,tier_b}.<method>.<metric>.{mean,std,n_seeds},
                                                label_robustness (second correctness labeler)
results/<backbone>/eval_extra[_test2].json      probe + judge fusion baselines (analysis.fusion)
results/<backbone>/tables.md, tables_test2.md   the main comparison, readable
results/paper/<backbone>/<d>_test.json          post-hoc calibrators on the final head (analysis.calibrators)
results/paper/<backbone>/<d>_signif.json        paired bootstrap vs. the strongest baseline (analysis.significance)
results/paper/bench_llama_hotpotqa.json         inference cost measurements (analysis.benchmark)
results/costs.json                              compute per stage (analysis.costs)
```

`tier_a` holds the methods without an external model at inference (including `chainuq-a`), `tier_b` the judge and
reward-model baselines and `chainuq`; `eval_extra*.json` adds the fusion baselines to `tier_b`. Metrics: `acc`, `auroc`, `prauc` (discrimination of
the raw score; the accuracy threshold is fit on validation) and `ece`, `brier`, `risk90` (calibration of the
calibrated probability); see [`metrics.py`](../chainuq/metrics.py).

## Analysis outputs

```text
$CHAINUQ_ANALYSIS/cache/<backbone>/<dataset>_<split>.pkl     head-independent evidence (analysis.cache)
$CHAINUQ_ANALYSIS/runs/<tag>.jsonl                           Stage-2 variant scores (analysis.variants)
$CHAINUQ_ANALYSIS/predictions/<variant>/<backbone>/*.npz     per-response probabilities (analysis.predictions)
$CHAINUQ_ANALYSIS/responses.json                             silent errors, AURC, fusion bootstrap, second labeler
$CHAINUQ_ANALYSIS/information/decomposition_<head>[_*].json  information decomposition (analysis.information)
$CHAINUQ_ANALYSIS/information/routing_<head>.json            judge routing (analysis.routing)
$CHAINUQ_ANALYSIS/information/babi_subtasks.json             bAbI within-task analysis (analysis.subtasks)
$CHAINUQ_ANALYSIS/information/unseen_domain.json             unseen-domain control (analysis.unseen)
```

A run record is `{"job": [variant, backbone, dataset, seed], "res": {split: {auroc, nll, ece, brier, risk90, prauc}}}`.
Decomposition files suffixed `_s3` use seeds 2026-2028; `_gbm` uses gradient boosting instead of logistic models;
`_<judge>` swaps the inference judge.

Only load trusted local PyTorch and pickle artifacts; feature caches and checkpoints are not an interchange format.
