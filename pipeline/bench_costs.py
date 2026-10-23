#!/usr/bin/env python3
"""Inference cost per response for the paper's efficiency tables (one GPU, HotpotQA test).

Measured here (wall clock, batched):
  heads        Stage-1 head forward on cached features (features moved to the GPU batch by batch);
               peak GPU memory of that inference.
  token        MCP / Perplexity / MTE from cached token log-probabilities (CPU).
  nli          CCP and the NLI part of Semantic Entropy / SelfCheckGPT (DeBERTa on the GPU).
  rm           reward-model forward.
  calibrators  fit on validation and predict on test, for PS / TS / ISO / BWH / R-Log (CPU).
LLM calls (greedy generation, 10 samples, CoT-UQ keyword call, judge calls) are amortized from the
stage logs aggregated by ``pipeline/costs.py`` (vLLM, batched, same GPU type).

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/bench_costs.py --backbone llama
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import calibration as C
from cuq.registry import FINAL_HEAD, results_dir

_here = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _module("train")
E = _module("evaluate")
S = _module("scores")
HEADS = ["saplma", "factoscope", "uhead", "meanpool-mlp", "attnpool-mlp", FINAL_HEAD]
N_NLI = 200


def timed(fn, repeat=3):
    best = None
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def bench_heads(backbone, ds, device):
    rows = {}
    for method in HEADS:
        keys = T.feature_keys_for(method)
        # Features already on the GPU, as when they come straight from the backbone's forward pass.
        samples = E.load_samples(backbone, ds, "test", feature_keys=keys, device=device)
        items = T.to_items(samples, device=device)
        model = T.load_trained(method, backbone, [2026], device)[2026]
        T.predict(model, items[:256], device)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        dt, _ = timed(lambda: T.predict(model, items, device, batch_size=64))
        rows[method] = {"ms_per_sample": 1000 * dt / len(items), "samples_per_s": len(items) / dt,
                        "peak_gpu_gb": (torch.cuda.max_memory_allocated(device) - base) / 1e9 + _param_gb(model),
                        "n": len(items)}
        if method == FINAL_HEAD:  # ChainUQ also scores every reasoning step and predicts the verdict
            def full():
                T.predict(model, items, device, batch_size=64)
                T.predict_steps(model, items, device)
                if hasattr(model, "forward_all"):
                    T.predict_verdict(model, items, device, batch_size=64)
            dt_full, _ = timed(full)
            rows[method]["ms_per_sample_full"] = 1000 * dt_full / len(items)
        del model, samples, items
        torch.cuda.empty_cache()
    # Lookback Lens: logistic regression on cached ratios (CPU).
    samples = E.load_samples(backbone, ds, "test", feature_keys=["lookback"])
    X = np.stack([s.feat["lookback"][0].float().numpy() for s in samples])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=200).fit(X[:500], np.arange(500) % 2)
    t0 = time.perf_counter()
    lr.decision_function(X)
    dt = time.perf_counter() - t0
    rows["lookback"] = {"ms_per_sample": 1000 * dt / len(X), "samples_per_s": len(X) / dt, "peak_gpu_gb": 0.0, "n": len(X)}
    return rows


def _param_gb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def bench_scores(backbone, ds, device):
    keys = ["ll", "ent", "concl_ids", "concl_topk_ids", "concl_topk_lp"]
    samples = E.load_samples(backbone, ds, "test", feature_keys=keys)
    t0 = time.perf_counter()
    S.score_token_methods(samples)
    dt = time.perf_counter() - t0
    rows = {m: {"ms_per_sample": 1000 * dt / len(samples) / 3, "peak_gpu_gb": 0.0} for m in ("mcp", "ppl", "mte")}
    from transformers import AutoTokenizer
    from cuq.nli import NLI
    from cuq.registry import BACKBONES
    tok = AutoTokenizer.from_pretrained(BACKBONES[backbone].path)
    nli = NLI(device)
    sub = samples[:N_NLI]
    torch.cuda.reset_peak_memory_stats(device)
    dt, _ = timed(lambda: S.score_ccp(sub, tok, nli), repeat=1)
    rows["ccp"] = {"ms_per_sample": 1000 * dt / len(sub), "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    dt, _ = timed(lambda: S.score_sampling(sub, backbone, ds, "test", "step", nli), repeat=1)
    rows["se_selfcheck_nli"] = {"ms_per_sample": 1000 * dt / len(sub)}
    from transformers import AutoModelForSequenceClassification
    from cuq.registry import RM_MODEL
    rm_tok = AutoTokenizer.from_pretrained(RM_MODEL.path)
    rm_tok.padding_side = "right"
    rm = AutoModelForSequenceClassification.from_pretrained(RM_MODEL.path, dtype=torch.bfloat16, num_labels=1).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    dt, _ = timed(lambda: S.score_rm(sub, rm, rm_tok, device), repeat=1)
    rows["rm"] = {"ms_per_sample": 1000 * dt / len(sub), "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    del rm
    torch.cuda.empty_cache()
    return rows, nli


def bench_calibrators(backbone, ds):
    hidden = json.loads((E.feat_dir(backbone, E.SOURCE_DATASET, "train") / "DONE.json").read_text())["hidden_size"]
    cell = E.Cell(backbone, ds, "step", hidden)
    yv = cell.labels("validation")
    pv, pt = E.load_preds(FINAL_HEAD, backbone, 2026, ds, "validation"), E.load_preds(FINAL_HEAD, backbone, 2026, ds, "test")
    sv, st = cell.scores("validation", pv), cell.scores("test", pt)
    J = E.cols(E.UF.GROUPS["u_judge"])
    t0 = time.perf_counter()
    Uv, Ut = cell.u_matrix("validation", pv), cell.u_matrix("test", pt)
    Xv = np.column_stack([Uv, cell.verdict("validation"), cell.extra("validation")])
    Xt = np.column_stack([Ut, cell.verdict("test"), cell.extra("test")])
    feat_ms = 1000 * (time.perf_counter() - t0) / (len(sv) + len(st))
    rows = {}
    for name, cls in (("ps", C.Platt), ("ts", C.Temperature), ("iso", C.Isotonic)):
        t0 = time.perf_counter(); m = cls().fit(sv, yv); fit = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(st); pred = time.perf_counter() - t0
        rows[name] = {"fit_s": fit, "ms_per_sample": 1000 * pred / len(st)}
    for name, cls in (("bwh", C.BWH), ("rlog", C.RLog2c)):
        t0 = time.perf_counter(); m = cls().fit(sv, Xv, yv); fit = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(st, Xt); pred = time.perf_counter() - t0
        rows[name] = {"fit_s": fit, "ms_per_sample": 1000 * pred / len(st) + feat_ms}
    rows["u_features_ms_per_sample"] = feat_ms
    rows["n_val"], rows["n_test"] = len(sv), len(st)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="llama")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--parts", default="heads,scores,calibrators", help="re-measure a subset; others are kept")
    args = p.parse_args()
    device = torch.device("cuda")
    path = results_dir() / "paper" / f"bench_{args.backbone}_{args.dataset}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    parts = args.parts.split(",")
    if "heads" in parts:
        out["heads"] = bench_heads(args.backbone, args.dataset, device)
    if "scores" in parts:
        out["scores"], _ = bench_scores(args.backbone, args.dataset, device)
    if "calibrators" in parts:
        out["calibrators"] = bench_calibrators(args.backbone, args.dataset)
    out["gpu"] = torch.cuda.get_device_name(device)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
