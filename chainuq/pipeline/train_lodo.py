"""Leave-one-domain-out imitation: for every OOD domain d, train the ChainUQ head on the labeled source and the
unlabeled pools of the other four domains only, and predict domain d (validation, test, test2) and HotpotQA.

Source responses and all pools are loaded once per backbone; runs are saved as ``<method>-lodo-<d>``.

    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.train_lodo --backbone llama --seeds 2026,2027,2028
"""

from __future__ import annotations

import argparse
import json

import torch

from chainuq import heads as H
from chainuq.artifacts import ckpt_dir
from chainuq.pipeline.train import head_dims, load_pool, predict_all
from chainuq.registry import BACKBONES, OOD_DATASETS, SOURCE_DATASET
from chainuq.samples import load_samples, load_verdicts
from chainuq.training import load_trained, save_trained, to_items, train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--targets", default=",".join(OOD_DATASETS), help="held-out domains")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--method", default="chainuq", help="a head with unlabeled pools")
    a = p.parse_args()
    bb, device, method = a.backbone, torch.device("cuda"), a.method
    spec = H.HEADS[method]
    assert spec.pool_size, f"{method} trains without unlabeled pools"
    keys = list(spec.features)
    seeds = [int(s) for s in a.seeds.split(",")]
    tr = load_samples(bb, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False, device=device)
    va = load_samples(bb, SOURCE_DATASET, "validation", feature_keys=keys, device=device)
    src = to_items(tr, device=device, verdicts=load_verdicts(bb, SOURCE_DATASET, "train"))
    val = to_items(va, device=device)
    del tr, va
    pools = {ds: load_pool(bb, ds, spec.pool_size, spec.pool_mode, keys, device) for ds in OOD_DATASETS}
    for d in a.targets.split(","):
        run = f"{method}-lodo-{d}"
        train_items = [dict(it) for it in src] + [it for ds, its in pools.items() if ds != d for it in its]
        dims = head_dims(bb, train_items)
        width = H.size_to_budget(method, dims)
        for seed in seeds:
            if (ckpt_dir(run, bb, seed) / "meta.json").exists():
                continue
            model, info = train_one(method, dims, width, train_items, [dict(it) for it in val], seed, device, a.epochs, a.batch_size, a.patience)
            info.update(dims=dims, n_train=len(train_items), n_val=len(val), held_out=d)
            save_trained(model, info, run, bb, seed)
            print(json.dumps({"run": run, "backbone": bb, "seed": seed, "train_seconds": info.get("train_seconds")}), flush=True)
        models = load_trained(run, bb, seeds, device, method)
        predict_all(run, method, bb, models, device, [SOURCE_DATASET, d])
        print(json.dumps({"predicted": run, "backbone": bb}), flush=True)


if __name__ == "__main__":
    main()
