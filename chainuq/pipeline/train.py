"""Stage 4: train Stage-1 heads on HotpotQA and write per-response predictions.

For every method and seed: train on HotpotQA train (label = the main judge's correctness label), early-stop on
HotpotQA validation PR-AUC, then predict the conclusion logit, one logit per reasoning step (span heads), and the
predicted chain verdict (PCS heads) on the validation and held-out splits of every dataset. See
``chainuq.training`` for the recipe and ``chainuq.heads.HEADS`` for the methods.

Examples:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.train --backbone llama --methods chainuq,uhead
    # label efficiency: correctness labels on 10% of the source responses (PCS targets stay on all)
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.train --backbone llama --methods chainuq \\
        --seeds 2026,2027,2028 --label_frac 0.1 --tag=-lf10
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from chainuq import heads as H
from chainuq.artifacts import ckpt_dir, feat_dir, feature_info, gen_path, pred_path, write_jsonl
from chainuq.config import HEAD_SEEDS
from chainuq.metrics import auroc
from chainuq.registry import BACKBONES, DATASETS, SOURCE_DATASET
from chainuq.samples import load_samples, load_verdicts
from chainuq.training import load_trained, predict, save_trained, to_items, train_one, write_predictions

EVAL_SPLITS = ("validation", "test", "test2")


def eval_cells(backbone, datasets, splits=EVAL_SPLITS):
    for ds in datasets:
        for split in splits:
            if (feat_dir(backbone, ds, split) / "DONE.json").exists() and gen_path(backbone, ds, split).exists():
                yield ds, split


def predict_all(run, method, backbone, models, device, datasets, force=False):
    """Predict every evaluation cell with every seed's model; each cell is loaded once, straight to the GPU."""
    spec = H.HEADS[method]
    for ds, split in eval_cells(backbone, datasets):
        todo = {sd: m for sd, m in models.items() if force or not pred_path(run, backbone, sd, ds, split).exists()}
        if not todo:
            continue
        items = to_items(load_samples(backbone, ds, split, feature_keys=list(spec.features), device=device), device=device)
        write_predictions(run, backbone, todo, items, ds, split, device, with_steps=spec.step_scores)
        del items
        torch.cuda.empty_cache()


def head_dims(backbone, train_items):
    info = feature_info(backbone)
    f = train_items[0]["feat"]
    return {"feature_dim": int(f["tok"].shape[1]) if "tok" in f else 0, "hidden_size": info["hidden_size"],
            "n_layers": info["n_layers"]}


def mask_labels(items, label_frac, keep_unlabeled):
    """Keep correctness labels on a fixed (md5-ordered) fraction of the source responses."""
    order = sorted(range(len(items)), key=lambda i: hashlib.md5(items[i]["uid"].encode()).hexdigest())
    keep = set(order[: int(round(label_frac * len(items)))])
    for i, it in enumerate(items):
        it["labeled"] = i in keep
    return items if keep_unlabeled else [it for it in items if it["labeled"]]


def select_saplma_layer(dims, train_items, val_items, seed, device, args):
    """SAPLMA probes one layer: pick it among {L/2, 2L/3, 3L/4, L} by validation AUROC."""
    L = dims["n_layers"]
    scores = {}
    for layer in sorted({L // 2, (2 * L) // 3, (3 * L) // 4, L}):
        d = dict(dims, saplma_layer=layer)
        m, _ = train_one("saplma", d, H.size_to_budget("saplma", d), train_items, val_items, seed, device,
                         args.epochs, args.batch_size, args.patience)
        scores[layer] = auroc([it["y"] for it in val_items], predict(m, val_items, device))
    return dict(dims, saplma_layer=max(scores, key=scores.get), saplma_layer_scores=scores)


def run_lookback(backbone, datasets, max_train):
    """Lookback Lens: logistic regression on response-level lookback ratios (C chosen on validation AUROC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    keys = list(H.LOOKBACK_FEATURES)
    tr = load_samples(backbone, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False)[: max_train or None]
    va = load_samples(backbone, SOURCE_DATASET, "validation", feature_keys=keys)
    X = lambda ss: np.stack([s.feat["lookback"][0].float().numpy() for s in ss])
    scaler = StandardScaler().fit(X(tr))
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X(tr)), [s.y for s in tr])
        a = auroc([s.y for s in va], clf.decision_function(scaler.transform(X(va))))
        if best is None or a > best[0]:
            best = (a, C, clf)
    seed = HEAD_SEEDS[0]
    for ds, split in eval_cells(backbone, datasets):
        ss = load_samples(backbone, ds, split, feature_keys=keys)
        logits = best[2].decision_function(scaler.transform(X(ss)))
        write_jsonl(pred_path("lookback", backbone, seed, ds, split),
                    [{"uid": s.uid, "logit": float(l), "step_logits": None} for s, l in zip(ss, logits)])
    out = ckpt_dir("lookback", backbone, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"C": best[1], "val_auroc": best[0], "n_params": int(best[2].coef_.size + 1)}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--methods", default="chainuq", help=f"comma-separated, from: lookback, {', '.join(H.HEADS)}")
    p.add_argument("--seeds", default="all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--eval_datasets", default=",".join(DATASETS))
    p.add_argument("--label_frac", type=float, default=1.0,
                   help="fraction of source responses with a correctness label; PCS heads keep their judge targets on all")
    p.add_argument("--tag", default="", help="suffix of the run name (checkpoints and predictions), e.g. -lf10")
    p.add_argument("--predict_only", action="store_true", help="reload checkpoints and rewrite the predictions")
    args = p.parse_args()
    datasets = [d for d in args.eval_datasets.split(",") if d]
    seeds = HEAD_SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda")
    bb = args.backbone

    for method in [m for m in args.methods.split(",") if m]:
        if method == "lookback":
            run_lookback(bb, datasets, args.max_train)
            continue
        spec, run = H.HEADS[method], method + args.tag
        if args.predict_only:
            predict_all(run, method, bb, load_trained(run, bb, seeds, device, method), device, datasets, force=True)
            continue
        todo = [sd for sd in seeds if not (ckpt_dir(run, bb, sd) / "meta.json").exists()]
        if todo:
            # Features go straight to the GPU, shard by shard, to bound host memory.
            keys = list(spec.features)
            tr = load_samples(bb, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False, device=device)
            va = load_samples(bb, SOURCE_DATASET, "validation", feature_keys=keys, device=device)
            verdicts = load_verdicts(bb, SOURCE_DATASET, "train") if spec.pcs else None
            train_items = to_items(tr[: args.max_train or None], device=device, verdicts=verdicts)
            val_items = to_items(va, device=device)
            del tr, va
            if args.label_frac < 1.0:  # without judge targets, unlabeled responses carry no training signal
                train_items = mask_labels(train_items, args.label_frac, keep_unlabeled=bool(spec.pcs))
            dims = head_dims(bb, train_items)
            if spec.arch == "saplma":
                dims = select_saplma_layer(dims, train_items, val_items, seeds[0], device, args)
            width = H.size_to_budget(method, dims)
            for seed in todo:
                model, info = train_one(method, dims, width, train_items, val_items, seed, device,
                                        args.epochs, args.batch_size, args.patience)
                info.update(dims=dims, n_train=len(train_items), n_val=len(val_items))
                save_trained(model, info, run, bb, seed)
                print(json.dumps({"run": run, "backbone": bb, "seed": seed,
                                  **{k: info[k] for k in ("n_params", "width", "best_val_prauc", "epochs_run", "train_seconds")}}),
                      flush=True)
            del train_items, val_items
            torch.cuda.empty_cache()
        # Also completes seeds whose earlier run stopped during prediction.
        models = load_trained(run, bb, seeds, device, method)
        predict_all(run, method, bb, models, device, datasets)
        print(json.dumps({"predicted": run, "backbone": bb, "seeds": sorted(models)}), flush=True)


if __name__ == "__main__":
    main()
