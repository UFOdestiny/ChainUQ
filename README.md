# CHAINUQ

CHAINUQ is a two-stage uncertainty pipeline for reasoning tasks with frozen LLMs.
It first trains a lightweight conclusion-focused uncertainty head on cached token
features, then refines the sample-level score with reasoning-aware post-hoc
calibration.

## Pipeline

```text
Generate -> Judge -> Cleanup -> Train -> Evaluate
```

1. `scripts/generate.py` runs backbone inference, extracts claims, and caches token features.
2. `scripts/judge.py` verifies conclusions and reasoning claims with an LLM judge when needed.
3. `scripts/cleanup_pending_claims.py` removes unresolved examples and rewrites caches into compact split storage.
4. `scripts/train.py` trains a lightweight uncertainty head on cached conclusion features.
5. `scripts/evaluate.py` evaluates the trained head and optionally applies post-hoc calibration.

## Included head family

The release keeps the paper-aligned CHAINUQ head and its ablations:

- `chainuq`
- `uq_abl_v1`
- `uq_abl_v2`
- `uq_abl_v3`
- `uq_abl_v4`

No external baseline implementations are included in this package.

## Layout

```text
config.py
data/
engine/
models/
  features/
  heads/
  wrapper.py
scripts/
utils/
requirements.txt
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate cached features:

```bash
python scripts/generate.py --dataset hotpotqa --split train,validation,test
```

Judge pending examples:

```bash
python scripts/judge.py --cache_dir artifacts/cached_features/<run>/<dataset>/<model> --split train,validation,test
```

Normalize caches:

```bash
python scripts/cleanup_pending_claims.py --cache_dir artifacts/cached_features/<run>/<dataset>/<model>
```

Train a head:

```bash
python scripts/train.py --head_type chainuq --cache_dir artifacts/cached_features/<run>/<dataset>/<model>
```

Evaluate with optional post-hoc calibration:

```bash
python scripts/evaluate.py \
  --head_path artifacts/results/<job>/train/chainuq/final_model \
  --cache_dir artifacts/cached_features/<run>/<dataset>/<model> \
  --split test \
  --enable_posthoc \
  --posthoc_method reasoning_logistic_isotonic
```

## Configuration

All runtime knobs live in `config.py` and can be overridden with environment
variables. By default, outputs are written under `artifacts/` inside this repo.
