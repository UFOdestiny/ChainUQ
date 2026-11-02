# Method

ChainUQ scores a chain-of-thought response `r = (step_1, ..., step_K, conclusion)` of a frozen backbone LLM with a
calibrated probability that its conclusion is correct. A head on the backbone's internal states learns to imitate an
answer-blind judge (Stage 1); a calibrator combines the head's outputs with chain evidence and, for ChainUQ, one
judge call (Stage 2). This page maps every part to its implementation.

## Responses and labels

Every prompt ends with a shared output contract (`Reasoning:` / `Step i:` / `Conclusion:`;
[`data/prompts.py`](../chainuq/data/prompts.py)). [`data/response.py`](../chainuq/data/response.py) parses a response
and aligns each step and the conclusion with a contiguous span of generated tokens; responses that break the
contract are dropped (`python -m chainuq.pipeline.audit` reports the rates).

- **Correctness `y`**: on HotpotQA, MuSiQue and bAbI the main judge (gpt-oss-20b) compares the conclusion with the
  reference answer (`correct` task); StrategyQA, GSM8K and MATH-500 use answer matching
  ([`data/answers.py`](../chainuq/data/answers.py)).
- **Judge outputs `J`**: the judge sees the task input and the response, never the reference answer
  ([`pipeline/judge.py`](../chainuq/pipeline/judge.py)). The `steps` task labels every reasoning step 0/1; the
  `final_blind` task returns a chain verdict with a verbalized probability `v` and P(true) from the token
  log-probabilities.

## Stage 1: imitation head

[`heads/chainuq.py`](../chainuq/heads/chainuq.py) reads cached features of one teacher-forced forward pass
([`pipeline/features.py`](../chainuq/pipeline/features.py)):

- **token branch** ([`heads/encoder.py`](../chainuq/heads/encoder.py)): per-token features are projected, the
  conclusion span is marked, a two-layer transformer contextualizes the response, and a span state (mean/max
  pooling, span-position statistics, claim-type embedding) is fused with a gated response summary;
- **layer branch**: all hidden layers at the token that predicts the conclusion and at the last conclusion token,
  mixed by learned softmax weights over layers;
- a sigmoid gate fuses the two states; branch dropout (p = 0.5) trains the output on each branch alone.

The answer output is trained with class-balanced BCE on `y` ([`training.py`](../chainuq/training.py)). Privileged
consistency supervision (PCS) adds the imitation of `J`: a categorical distribution `q(L | A)` over the four levels
of the verbalized verdict (edges 0.25 / 0.85 / 0.97) and a score for each reasoning step, read from the token-branch
state of the step span marked as a reasoning claim. Both are trained by their negative log-likelihood, so the PCS
loss is the imitation cross-entropy `E[-log q(J | A)]`.

The judge outputs need no reference answer, so PCS also runs on unlabeled questions. The final head (`chainuq`)
trains on labeled HotpotQA train and on unlabeled train pools of the five OOD domains (6,000 questions per domain;
all 1,443 StrategyQA train questions), which contribute the PCS loss only. Every neural head, including the probe
baselines, is sized to the largest width whose trainable parameters stay within 2M.

## Stage 2: calibrator

The evidence of an evaluated response ([`evidence.py`](../chainuq/evidence.py),
[`evaluation.py`](../chainuq/evaluation.py)) is built from the head and the generated text only:

- the head logit `s` (`u_base`), the chain structure and the reasoning-conclusion feature gap (`u_gap`);
- token uncertainty of the conclusion span and of the weakest reasoning step (span features);
- the head's predicted verdict and its imitation distribution `q(L | A)`;
- the score of a second internal-state probe (SAPLMA, `registry.EXTRA_PROBES`).

| Variant | Additional evidence | External model at inference |
|---|---|---|
| **ChainUQ-A** | none | none |
| **ChainUQ** | the judge's verdict `v` and log-odds of P(true) (`final_blind`) | one judge call |

R-Log ([`calibration.py`](../chainuq/calibration.py)) chooses its capacity by inner 5-fold log-loss on the fitting
split: isotonic regression on `s`, regularized logistic regression, cross-fitted reasoning-conditioned logistic
mixtures, shallow gradient boosting, and a compact logistic model on `s` and the judge and imitation outputs
(evidence-set selection, `RLogSel`). Every calibrator and decision threshold is fit on the validation split of the
evaluated dataset and applied once to its held-out split, so target-domain calibration uses correctness-labeled
validation data; evidence construction never reads the correctness label.

## Information decomposition and judge routing

[`analysis/information.py`](../chainuq/analysis/information.py) estimates, with logistic models fit on validation
and scored on test, how much the judge-free evidence `A` and the judge output `J` know about `y` beyond each other:

```text
I(y; A | J) ~ CE(y | J) - CE(y | A, J)        what the internal states add to the judge
I(y; J | A) ~ CE(y | A) - CE(y | A, J)        what a judge call adds to the internal states
CE(y | J) - CE(y | A) = I(y; A | J) - I(y; J | A)
```

The imitation loss `CE(L | A)` bounds the residual `I(y; J | A)`; the same file computes the bound and its calibrated
forms. [`analysis/routing.py`](../chainuq/analysis/routing.py) spends a budget of judge calls on the responses with
the largest expected information gain `EIG(a) = H(sum_j q(j|a) p_j) - sum_j q(j|a) H(p_j)`, where `p_j` is
ChainUQ with the judge output set to level `j`.

## Stage-2 variants

[`analysis/variants.py`](../chainuq/analysis/variants.py) scores any (evidence, calibrator, head) combination on the
cached evidence ([`analysis/cache.py`](../chainuq/analysis/cache.py)). A variant is written
`<blockset>[+block][-block][@calibrator][#head]`:

```text
B-u_reason-u_token-u_judge+ptrue+head:saplma+vq@rlogs#chainuq     ChainUQ
A-u_reason-u_token+head:saplma+vq@rlogs#chainuq                   ChainUQ-A
S+verdict+ptrue@lrsel#saplma                                      SAPLMA + judge fusion baseline
S@iso#uhead                                                       UHead with isotonic regression
```

Blocksets are `B` (with the judge), `A` (judge-free) and `S` (head score only); blocks are `u_base`, `u_reason`,
`u_token`, `u_gap`, `u_judge`, `span`, `vhat`, `vq`, `verdict`, `ptrue` and `head:<run>` (another head's score);
calibrators are `rlogs`, `rlog`, `lrsel`, `gbm` and `iso`. `pipeline.evaluate` and the variant runner implement the
same final method (`evaluation.final_evidence`, `evaluation.final_calibrator`).
