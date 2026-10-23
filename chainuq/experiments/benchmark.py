"""Inference cost per response for the efficiency tables (one GPU, HotpotQA test).

Measured here (wall clock, batched):
  heads        head forward on cached features (moved to the GPU batch by batch) and its peak GPU memory;
               ChainUQ additionally scores every reasoning step and predicts the verdict
  scores       MCP / Perplexity / MTE from cached log-probabilities (CPU), CCP and the NLI part of Semantic
               Entropy / SelfCheckGPT (DeBERTa), and the reward model
  calibrators  fit on validation and predict on test for PS / TS / ISO / BWH / R-Log (CPU)
LLM calls (greedy generation, 10 samples, CoT-UQ keywords, judge calls) are amortized from the stage statistics
aggregated by ``experiments.costs``. Writes ``results/paper/bench_<backbone>_<dataset>.json``.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.experiments.benchmark --backbone llama
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from chainuq import calibration as C
from chainuq import heads as H
from chainuq.artifacts import feature_info, results_dir
from chainuq.evaluation import Cell, HeadEvidence, load_preds
from chainuq.pipeline import scores as S
from chainuq.registry import BACKBONES, FINAL_HEAD, RM_MODEL
from chainuq.samples import load_samples
from chainuq.training import load_trained, predict, predict_steps, predict_verdict, to_items

HEADS = ["saplma", "factoscope", "uhead", "meanpool-mlp", "attnpool-mlp", FINAL_HEAD]
N_NLI = 200


def timed(fn, repeat=3):
    best, out = None, None
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def _param_gb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def bench_head(method, backbone, ds, device):
    # Features already on the GPU, as when they come straight from the backbone's forward pass.
    items = to_items(load_samples(backbone, ds, "test", feature_keys=list(H.HEADS[method].features), device=device),
                     device=device)
    model = load_trained(method, backbone, [2026], device)[2026]
    predict(model, items[:256], device)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    dt, _ = timed(lambda: predict(model, items, device, batch_size=64))
    row = {"ms_per_sample": 1000 * dt / len(items), "samples_per_s": len(items) / dt,
           "peak_gpu_gb": (torch.cuda.max_memory_allocated(device) - base) / 1e9 + _param_gb(model), "n": len(items)}
    if method == FINAL_HEAD:  # ChainUQ also scores every reasoning step and predicts the verdict
        def full():
            predict(model, items, device, batch_size=64)
            predict_steps(model, items, device)
            predict_verdict(model, items, device, batch_size=64)
        row["ms_per_sample_full"] = 1000 * timed(full)[0] / len(items)
    return row


def bench_heads(backbone, ds, device):
    rows = {}
    for method in HEADS:
        rows[method] = bench_head(method, backbone, ds, device)
        torch.cuda.empty_cache()
    # Lookback Lens: logistic regression on cached ratios (CPU).
    from sklearn.linear_model import LogisticRegression
    X = np.stack([s.feat["lookback"][0].float().numpy() for s in load_samples(backbone, ds, "test", feature_keys=["lookback"])])
    lr = LogisticRegression(max_iter=200).fit(X[:500], np.arange(500) % 2)
    t0 = time.perf_counter()
    lr.decision_function(X)
    dt = time.perf_counter() - t0
    rows["lookback"] = {"ms_per_sample": 1000 * dt / len(X), "samples_per_s": len(X) / dt, "peak_gpu_gb": 0.0, "n": len(X)}
    return rows


def bench_scores(backbone, ds, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from chainuq.nli import NLI

    samples = load_samples(backbone, ds, "test", feature_keys=["ll", "ent", "concl_ids", "concl_topk_ids", "concl_topk_lp"])
    t0 = time.perf_counter()
    S.score_token_methods(samples)
    dt = time.perf_counter() - t0
    rows = {m: {"ms_per_sample": 1000 * dt / len(samples) / 3, "peak_gpu_gb": 0.0} for m in ("mcp", "ppl", "mte")}
    tok = AutoTokenizer.from_pretrained(BACKBONES[backbone].path)
    nli = NLI(device)
    sub = samples[:N_NLI]
    torch.cuda.reset_peak_memory_stats(device)
    dt, _ = timed(lambda: S.score_ccp(sub, tok, nli), repeat=1)
    rows["ccp"] = {"ms_per_sample": 1000 * dt / len(sub), "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    dt, _ = timed(lambda: S.score_sampling(sub, backbone, ds, "test", nli), repeat=1)
    rows["se_selfcheck_nli"] = {"ms_per_sample": 1000 * dt / len(sub)}
    rm_tok = AutoTokenizer.from_pretrained(RM_MODEL.path)
    rm_tok.padding_side = "right"
    rm = AutoModelForSequenceClassification.from_pretrained(RM_MODEL.path, dtype=torch.bfloat16, num_labels=1).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    dt, _ = timed(lambda: S.score_rm(sub, rm, rm_tok, device), repeat=1)
    rows["rm"] = {"ms_per_sample": 1000 * dt / len(sub), "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    return rows


def bench_calibrators(backbone, ds):
    cell = Cell(backbone, ds, "test", feature_info(backbone)["hidden_size"])
    yv = cell.labels("validation")
    pv, pt = load_preds(FINAL_HEAD, backbone, 2026, ds, "validation"), load_preds(FINAL_HEAD, backbone, 2026, ds, "test")
    t0 = time.perf_counter()
    ev = HeadEvidence(cell, pv, pt, use_predicted_verdict=False)
    Xv, Xt = ev.matrices("B")
    feat_ms = 1000 * (time.perf_counter() - t0) / (len(ev.sv) + len(ev.st))
    rows = {}
    for name, cls in (("ps", C.Platt), ("ts", C.Temperature), ("iso", C.Isotonic)):
        t0 = time.perf_counter(); m = cls().fit(ev.sv, yv); fit = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(ev.st); pred = time.perf_counter() - t0
        rows[name] = {"fit_s": fit, "ms_per_sample": 1000 * pred / len(ev.st)}
    for name, cls in (("bwh", C.BWH), ("rlog", C.RLog)):
        t0 = time.perf_counter(); m = cls().fit(ev.sv, Xv, yv); fit = time.perf_counter() - t0
        t0 = time.perf_counter(); m.predict(ev.st, Xt); pred = time.perf_counter() - t0
        rows[name] = {"fit_s": fit, "ms_per_sample": 1000 * pred / len(ev.st) + feat_ms}
    rows["u_features_ms_per_sample"] = feat_ms
    rows["n_val"], rows["n_test"] = len(ev.sv), len(ev.st)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="llama")
    p.add_argument("--dataset", default="hotpotqa")
    p.add_argument("--parts", default="heads,scores,calibrators", help="re-measure a subset; the others are kept")
    args = p.parse_args()
    device = torch.device("cuda")
    path = results_dir() / "paper" / f"bench_{args.backbone}_{args.dataset}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    parts = args.parts.split(",")
    if "heads" in parts:
        out["heads"] = bench_heads(args.backbone, args.dataset, device)
    if "scores" in parts:
        out["scores"] = bench_scores(args.backbone, args.dataset, device)
    if "calibrators" in parts:
        out["calibrators"] = bench_calibrators(args.backbone, args.dataset)
    out["gpu"] = torch.cuda.get_device_name(device)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
