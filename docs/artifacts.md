# Artifacts

Every stage reads and writes under `$CHAINUQ_ARTIFACTS` through the path functions of
[`chainuq/artifacts.py`](../chainuq/artifacts.py). Splits are `train`, `validation`, `test`, and `test2`.

```text
gen/<backbone>/<dataset>/<split>.jsonl                      greedy generations
gen/<backbone>/<dataset>/<split>.samples.jsonl              10 temperature samples (Semantic Entropy, SelfCheckGPT)
gen/<backbone>/<dataset>/<split>.stats.json                 counts and timing
judge/<judge>/<task>/<backbone>/<dataset>/<split>.jsonl     judge annotations (+ .stats.json)
feat/<backbone>/<dataset>/<split>/shard_*.pt                cached features (+ DONE.json with backbone dimensions)
cotuq/<backbone>/<dataset>/<split>.jsonl                    CoT-UQ keywords
ckpt/<method>/<backbone>/seed<k>/model.pt, meta.json        trained heads and their training summary
preds/<method>/<backbone>/seed<k>/<dataset>/<split>.jsonl   per-response scores of every method
results/                                                    evaluation outputs (below)
```

## Record formats

**Generation** (`gen/`): `uid`, `question`, `user_content`, `gold`, `gold_aliases`, `n_hops`, `prompt_ids`,
`gen_ids`, `gen_text`, `finish_reason`, `pred_answer`, `label_em` (answer-matching datasets), `parse_ok`,
`drop_reason`, and, if parsed, `steps` / `conclusion` as `{"text", "tok": [start, end)}` in generated-token
coordinates.

**Judge** (`judge/`): `uid`, `value` (0/1 for `correct` and `final_blind`, a 0/1 list per step for `steps`
and `steps_noconcl`, `null` if the output did not parse), `raw`, token counts; `final_blind` also has
`p_verbal` (verbalized probability) and `p_true` (P(true) from token log-probabilities). Judges: `gptoss`
(main), `qwen4b`, `qwen1b`, `self-<backbone>`, `nli` (DeBERTa entailment).

**Features** (`feat/`): a dict `uid -> tensors`; see the docstring of
[`pipeline/features.py`](../chainuq/pipeline/features.py).

**Predictions** (`preds/`): `uid`, `logit` (raw confidence score; higher = more likely correct),
`step_logits` (one score per reasoning step, or `null`), and `verdict_hat` (predicted chain verdict, PCS heads).

## Method keys

| Key | Meaning |
|---|---|
| `chainuq` | ChainUQ head with privileged consistency supervision (PCS) |
| `chainuq-wo-step`, `chainuq-wo-verdict` | PCS without the step-label or verdict target |
| `chainuq-wo-pcs` | head trained on correctness labels only |
| `chainuq-wo-pcs-layer`, `-token`, `-bdrop` | without PCS and without the layer branch / token branch / branch dropout |
| `chainuq-lf10`, `chainuq-lf25`, `chainuq-wo-pcs-lf10`, `-lf25` | correctness labels on 10% / 25% of HotpotQA train |
| `chainuq-ta` | label-free target adaptation (consistency targets of the main judge) |
| `chainuq-ta-src`, `-pseudo`, `-steps`, `-both` | controls: no target loss, verdict as pseudo-label, step labels only, both |
| `chainuq-ta-mid[-pseudo]`, `chainuq-ta-weak[-pseudo]` | pool labeled by Qwen3-4B / Qwen3-1.7B |
| `chainuq-ta-n<N>` | pool of N target questions |
| `uhead-pcs` | UHead trained with the same PCS targets |
| `mcp`, `ppl`, `mte`, `ccp`, `cotuq`, `se`, `selfcheck` | judge-free baselines without training |
| `saplma`, `factoscope`, `lookback`, `uhead`, `meanpool-mlp`, `attnpool-mlp` | supervised probe baselines |
| `judge-step`, `judge-final`, `rm` | judge and reward-model baselines |

In `results/<backbone>/eval*.json`, `tier_a` holds methods without an external model at inference (including
`chainuq-a`) and `tier_b` the judge and reward-model baselines and `chainuq`.

## Results

```text
results/<backbone>/eval.json, eval_test2.json   main comparison: datasets.<d>.{tier_a,tier_b}.<method>.<metric>.{mean,std,n_seeds},
                                                judge_agreement, label_robustness
results/<backbone>/tables.md, tables_test2.md   the same, readable
results/paper/<backbone>/<d>_test.json          calibrators, stage2, judge_swap, choice, heads_raw/heads_B/heads_A
results/paper/<backbone>/<d>_test_cases.json    per-response scores for the case studies (Llama)
results/paper/<backbone>/<d>_signif.json        paired bootstrap vs. the strongest baseline
results/paper/<backbone>/<d>_adapt[_<tag>].json label-free adaptation: {A,B}_{test,test2}.{base,adapted,selected}
results/paper/bench_llama_hotpotqa.json         inference cost measurements
results/costs.json                              compute per stage
results/paper/summary.json                      win counts and other summary numbers of the paper
```

Metrics: `acc`, `auroc`, `prauc` (discrimination of the raw score; the accuracy threshold is fit on validation)
and `ece`, `brier`, `risk90` (calibration of the calibrated probability); see
[`chainuq/metrics.py`](../chainuq/metrics.py).
