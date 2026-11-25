#!/usr/bin/env python3
"""Write the paper's result tables (LaTeX) and summary numbers from the evaluation outputs.

Inputs: ``results/<backbone>/eval.json`` and ``eval_test2.json`` (pipeline/evaluate.py) and
``results/paper/<backbone>/<dataset>_test.json`` (pipeline/paper_tables.py).
Outputs: ``<latex>/table/*.tex`` and ``results/paper/summary.json``.

Example:
    python pipeline/paper_latex.py --latex ../latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.registry import FINAL_HEAD, results_dir

BACKBONES = [("llama", "Llama-3.1-8B-Instruct"), ("qwen8b", "Qwen3-8B"), ("ministral", "Ministral-8B-Instruct-2410")]
DATASETS = [("hotpotqa", "HotpotQA (ID)"), ("musique", "MuSiQue"), ("strategyqa", "StrategyQA"), ("babi", "bAbI"),
            ("gsm8k", "GSM8K"), ("math500", "MATH-500")]
LOWER = {"ece", "brier", "risk90"}
MNAME = {"acc": "Acc", "auroc": "AUROC", "prauc": "PRAUC", "ece": "ECE", "brier": "Brier", "risk90": "Risk@90"}

# (group label, [(key, tier, display name)]) -- the methods of the main comparison.
GROUPS = [
    ("Single-response, unsupervised", [("mcp", "tier_a", "MCP"), ("ppl", "tier_a", "Perplexity"),
                                       ("mte", "tier_a", "MTE"), ("ccp", "tier_a", "CCP")]),
    ("Extra backbone calls", [("cotuq", "tier_a", "CoT-UQ"), ("se", "tier_a", "Semantic Entropy"),
                              ("selfcheck", "tier_a", "SelfCheckGPT")]),
    ("Supervised probes", [("saplma", "tier_a", "SAPLMA"), ("factoscope", "tier_a", "Factoscope"),
                           ("lookback", "tier_a", "Lookback Lens"), ("uhead", "tier_a", "UHead"),
                           ("meanpool-mlp", "tier_a", "MeanPool-MLP"), ("attnpool-mlp", "tier_a", "AttnPool-MLP")]),
    ("OURS_A", [("chainuq-v3-A", "tier_a", r"\textbf{\mA}")]),
    ("External verifier", [("judge_step", "tier_b", "Judge-Step"), ("judge_final", "tier_b", "Judge-Final"),
                           ("rm", "tier_b", "Reward Model")]),
    ("OURS", [("chainuq-v3", "tier_b", r"\textbf{\m}")]),
]
OURS, OURS_A = "chainuq-v3", "chainuq-v3-A"
BASELINES = [k for g, ms in GROUPS for k, _, _ in ms if not g.startswith("OURS")]
TIER_A_BASELINES = [k for g, ms in GROUPS if g in ("Single-response, unsupervised", "Extra backbone calls",
                                                    "Supervised probes") for k, _, _ in ms]


def load(bb, split="test"):
    return json.loads((results_dir() / bb / ("eval.json" if split == "test" else f"eval_{split}.json")).read_text())


def paper_json(bb, ds):
    p = results_dir() / "paper" / bb / f"{ds}_test.json"
    return json.loads(p.read_text()) if p.exists() else None


def value(R, key, tier, metric):
    v = R[tier].get(key, {}).get(metric)
    return None if v is None else v["mean"]


def rank_marks(vals, metric):
    """Bold the best and underline the second best of a column (ties share the mark)."""
    ok = [v for v in vals if v is not None]
    order = sorted(set(round(v, 3) for v in ok), reverse=metric not in LOWER)
    best = order[0] if order else None
    second = order[1] if len(order) > 1 else None

    def fmt(v):
        if v is None:
            return "--"
        r = round(v, 3)
        s = f"{r:.3f}"
        return rf"\textbf{{{s}}}" if r == best else (rf"\underline{{{s}}}" if r == second else s)
    return [fmt(v) for v in vals]


def header(metrics, datasets, first="Method"):
    n = len(metrics)
    cols = "l" + (" " + "c" * n) * len(datasets)
    top = " & ".join([""] + [rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{name}}}}}" for _, name in datasets]) + r" \\"
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(datasets)))
    arrows = [MNAME[m] + (r"$\downarrow$" if m in LOWER else r"$\uparrow$") for m in metrics]
    sub = " & ".join([rf"\textbf{{{first}}}"] + arrows * len(datasets)) + r" \\"
    return cols, [r"\toprule", top, rules, sub, r"\midrule"]


def method_table(bb, metrics, caption, label, star=True):
    res = load(bb)["datasets"]
    rows = [(g, k, t, name) for g, ms in GROUPS for k, t, name in ms]
    cells = {}
    for ds, _ in DATASETS:
        for m in metrics:
            vals = [value(res[ds], k, t, m) if ds in res else None for _, k, t, _ in rows]
            for (_, k, _, _), s in zip(rows, rank_marks(vals, m)):
                cells[(k, ds, m)] = s
    cols, lines = header(metrics, DATASETS)
    prev = None
    for g, k, t, name in rows:
        if g != prev and prev is not None:
            lines.append(r"\midrule")
        pass  # groups are separated by rules; their names are given in the caption
        prev = g
        body = " & ".join(cells[(k, ds, m)] for ds, _ in DATASETS for m in metrics)
        shade = r"\rowcolor{gray!20} " if g.startswith("OURS") else ""
        lines.append(f"{shade}{name} & {body} \\\\")
    lines.append(r"\bottomrule")
    env = "table*" if star else "table"
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", r"\setlength{\tabcolsep}{2pt}",
                      r"\begin{adjustbox}{max width=\textwidth, keepaspectratio}", rf"\begin{{tabular}}{{{cols}}}",
                      *lines, r"\end{tabular}", r"\end{adjustbox}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      rf"\end{{{env}}}", ""])


CALIBRATORS = [("ps", "PS"), ("ts", "TS"), ("iso", "ISO"), ("bwh", "BWH"), ("rlog", r"\textbf{R-Log}")]


def calibrator_table(bb, caption, label):
    metrics = ["ece", "brier", "risk90"]
    cells = {}
    for ds, _ in DATASETS:
        P = paper_json(bb, ds)
        for m in metrics:
            vals = [P["calibrators"][k][m]["mean"] if P and k in P.get("calibrators", {}) else None for k, _ in CALIBRATORS]
            for (k, _), s in zip(CALIBRATORS, rank_marks(vals, m)):
                cells[(k, ds, m)] = s
    cols, lines = header(metrics, DATASETS)
    for k, name in CALIBRATORS:
        shade = r"\rowcolor{gray!20} " if k == "rlog" else ""
        lines.append(f"{shade}{name} & " + " & ".join(cells[(k, ds, m)] for ds, _ in DATASETS for m in metrics) + r" \\")
    lines.append(r"\bottomrule")
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{4pt}",
                      r"\begin{adjustbox}{max width=\textwidth, keepaspectratio}", rf"\begin{{tabular}}{{{cols}}}",
                      *lines, r"\end{tabular}", r"\end{adjustbox}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{table*}", ""])


def wins(split="test"):
    """Cells where ours >= the strongest baseline of the main comparison (tier rules as in the tables)."""
    out = {}
    for tier, ours, pool in (("B", OURS, BASELINES), ("A", OURS_A, TIER_A_BASELINES)):
        per = {m: [0, 0] for m in MNAME}
        gains = []
        for bb, _ in BACKBONES:
            res = load(bb, split)["datasets"]
            for ds, R in res.items():
                for m in MNAME:
                    tier_of = {k: t for _, ms in GROUPS for k, t, _ in ms}
                    o = value(R, ours, tier_of[ours], m)
                    cand = [value(R, k, tier_of[k], m) for k in pool]
                    cand = [c for c in cand if c is not None]
                    if o is None or not cand:
                        continue
                    best = min(cand) if m in LOWER else max(cand)
                    per[m][0] += (o <= best) if m in LOWER else (o >= best)
                    per[m][1] += 1
                    if m == "auroc":
                        gains.append((o - best) / best)
        out[tier] = {"per_metric": per, "total": [sum(a for a, _ in per.values()), sum(b for _, b in per.values())],
                     "mean_rel_auroc_gain": float(np.mean(gains)), "max_rel_auroc_gain": float(np.max(gains))}
    return out


def fmt3(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.3f}"


def wrap_table(lines, cols, caption, label, star=False, size=None, sep="3pt"):
    env = "table*" if star else "table"
    width = r"\textwidth" if star else r"\linewidth"
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", rf"\setlength{{\tabcolsep}}{{{sep}}}"] + ([size] if size else []) +
                     [rf"\begin{{adjustbox}}{{max width={width}, keepaspectratio}}", rf"\begin{{tabular}}{{{cols}}}",
                      *lines, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\caption{{{caption}}}",
                      rf"\label{{{label}}}", rf"\end{{{env}}}", ""])


def agreement_table():
    rows = [("Correctness: Qwen3-4B", "label", None), ("Steps: Qwen3-1.7B", "step", "qwen1b"),
            ("Steps: backbone itself", "step", "self"), ("Steps: DeBERTa NLI", "step", "nli")]
    cols, lines = "l" + " cc" * len(DATASETS), [r"\toprule"]
    lines.append(" & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{{n.replace(' (ID)', '')}}}}}" for _, n in DATASETS]) + r" \\")
    lines.append(" ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))))
    lines.append(" & ".join([r"\textbf{Labeler vs.\ main judge}"] + ["Agr.", r"$\kappa$"] * len(DATASETS)) + r" \\")
    lines.append(r"\midrule")
    stats = {}
    for name, kind, key in rows:
        cells = []
        for ds, _ in DATASETS:
            ag, ka = [], []
            for bb, _ in BACKBONES:
                R = load(bb)["datasets"][ds]
                if kind == "label":
                    lr = R.get("label_robustness") or {}
                    if lr.get("label_kappa") is not None:
                        ag.append(lr["label_agreement"]); ka.append(lr["label_kappa"])
                else:
                    j = R.get("judge_agreement", {}).get(f"self-{bb}" if key == "self" else key)
                    if j:
                        ag.append(j["step_agreement"]); ka.append(j["step_kappa"])
            stats[(name, ds)] = (ka, ag)
            cells += [f"{100 * np.mean(ag):.1f}", f"{np.mean(ka):.2f}"] if ag else ["--", "--"]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lab = [k for (n, _), (ks, _) in stats.items() if n.startswith("Corr") for k in ks]
    stp = [k for (n, _), (ks, _) in stats.items() if n.startswith("Steps") for k in ks]
    tex = wrap_table(lines, cols, "Agreement (\\%) and Cohen's $\\kappa$ of alternative labelers with the main judge (gpt-oss-20b) on the test splits, averaged over the three backbones. Correctness is compared only where it is judge-labeled.",
                     "tab:judge_agreement", star=True)
    return tex, {"label_kappa_min": min(lab), "label_kappa_max": max(lab), "step_kappa_max": max(stp)}


VERIFIERS = [("gptoss", "gpt-oss-20b (main)"), ("gptoss-noconcl", "gpt-oss-20b, conclusion withheld"),
             ("qwen1b", "Qwen3-1.7B"), ("self", "Backbone itself"), ("nli", "DeBERTa NLI")]


def judge_swap_table():
    cols, lines = "l" + "c" * (len(BACKBONES) + 1), [r"\toprule"]
    lines.append(" & ".join([r"\textbf{Step-label source}"] + [n.split("-")[0] for _, n in BACKBONES] + ["Mean"]) + r" \\")
    lines.append(r"\midrule")
    means = {}
    for key, name in VERIFIERS + [("none", r"None (\mA)")]:
        per = []
        for bb, _ in BACKBONES:
            vals = []
            for ds, _ in DATASETS:
                P = paper_json(bb, ds)
                if not P:
                    continue
                src = P["calibrators"].get("rlog_A") if key == "none" else P.get("judge_swap", {}).get(key)
                if src:
                    vals.append(src["auroc"]["mean"])
            per.append(np.mean(vals) if vals else np.nan)
        means[key] = float(np.nanmean(per))
        lines.append(f"{name} & " + " & ".join(fmt3(v) for v in per) + f" & {fmt3(means[key])} \\\\")
    tex = wrap_table(lines, cols, "AUROC of \\m without the chain-level verdict when the step labels come from different verifiers (mean over six datasets).",
                     "tab:judge_swap")
    return tex, means


def calibrators_for(bb):
    return calibrator_table(bb, f"Post-hoc calibrators applied to the same \\m head score of {dict(BACKBONES)[bb]}.",
                            "tab:calibration" if bb == "llama" else f"tab:calibration_{[b for b, _ in BACKBONES].index(bb) + 1}")


STAGE1_ROWS = {
    "chainuq2bd": [("-LB", "chainuq2bd-lb"), ("-CS", "chainuq2bd-cs"), ("-CF", "chainuq2bd-cf"), ("-FR", "chainuq2bd-fr"),
                   ("- Layer branch", "chainuq2bd-tok"), ("- Token branch", "chainuq2bd-lay"), ("- Branch dropout", "chainuq2")],
    "chainuq2bd-core": [("+ Refinement block", "chainuq2bd"), ("- Layer branch", "chainuq2bd-core-tok"),
                        ("- Token branch", "chainuq2bd-lay"), ("- Branch dropout", "chainuq2-core")],
    "chainuq2pcs": [("- Step supervision", "chainuq2pcs-verdict"), ("- Verdict supervision", "chainuq2pcs-steps"),
                    ("- Both (no PCS)", "chainuq2bd-core"), ("- Layer branch$^\\dagger$", "chainuq2bd-core-tok"),
                    ("- Token branch$^\\dagger$", "chainuq2bd-lay"), ("- Branch dropout$^\\dagger$", "chainuq2-core")],
}[FINAL_HEAD]
ABL_ROWS = [("Full \\m", "calibrators", "rlog"),
            ("Stage 1", None, None)] + [(l, "heads_B", k) for l, k in STAGE1_ROWS] + [
            ("Head $\\rightarrow$ UHead", "heads_B", "uhead"), ("Head $\\rightarrow$ SAPLMA", "heads_B", "saplma"),
            ("Stage 2", None, None),
            ("- Step judgments", "stage2", "-u_judge"), ("- Chain verdict $v(r)$", "stage2", "-verdict"),
            ("- Span token features", "stage2", "-token_extra"), ("Fixed capacity", "stage2", "fixed_capacity")]


def ablation_table(bb="llama"):
    """Ablations averaged over the six datasets, for full ChainUQ and for the judge-free ChainUQ-A."""
    metrics = ["auroc", "ece", "brier", "risk90"]
    tiers = (("B", r"\m (with judge)"), ("A", r"\mA (judge-free)"))
    src = {"B": {"calibrators": ("calibrators", "rlog"), "heads_B": "heads_B", "stage2": "stage2"},
           "A": {"calibrators": ("calibrators", "rlog_A"), "heads_B": "heads_A", "stage2": None}}
    vals = {}
    for label, sec, key in ABL_ROWS:
        if not sec:
            continue
        for tier, _ in tiers:
            for m in metrics:
                xs = []
                for ds, _ in DATASETS:
                    P = paper_json(bb, ds) or {}
                    if sec == "calibrators":
                        v = P.get("calibrators", {}).get(src[tier]["calibrators"][1], {}).get(m)
                    elif src[tier][sec] is None:
                        v = None
                    else:
                        v = P.get(src[tier][sec], {}).get(key, {}).get(m)
                    xs.append(None if v is None else v["mean"])
                vals[(label, tier, m)] = float(np.mean(xs)) if xs and all(x is not None for x in xs) else None
    cols, lines = header(metrics, [(t, n) for t, n in tiers], first="Variant")
    full = ABL_ROWS[0][0]
    for label, sec, key in ABL_ROWS:
        if sec is None:
            lines.append(r"\midrule")
            continue
        cells = []
        for tier, _ in tiers:
            for m in metrics:
                v = vals[(label, tier, m)]
                cells.append(rf"\textbf{{{fmt3(v)}}}" if label == full else fmt3(v))
        shade = r"\rowcolor{gray!20} " if label == full else ""
        lines.append(f"{shade}{label} & " + " & ".join(cells) + r" \\")
    cap = (f"Ablation of both stages of \\m on {dict(BACKBONES)[bb]}, averaged over the six datasets (10 seeds); rows grouped into Stage 1 and Stage 2. "
           r"$^\dagger$: architecture ablations trained without PCS. Stage-2 judge ablations do not apply to \mA.")
    return wrap_table(lines, cols, cap, "tab:ablation", star=False, sep="1.5pt"), vals


def ablation_all_table():
    """Head ablations (full pipeline and judge-free) for every backbone, AUROC averaged over the six datasets."""
    rows = [("Full \\m", None)] + [(l, k) for l, sec, k in ABL_ROWS if sec == "heads_B"]
    cols = "l" + " cc" * len(BACKBONES)
    lines = [r"\toprule", " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{{n.split('-')[0]}}}}}" for _, n in BACKBONES]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BACKBONES))),
             " & ".join([r"\textbf{Variant}"] + [r"\m", r"\mA"] * len(BACKBONES)) + r" \\", r"\midrule"]
    for label, key in rows:
        cells = []
        for bb, _ in BACKBONES:
            for sec, full in (("heads_B", "rlog"), ("heads_A", "rlog_A")):
                vals = []
                for ds, _ in DATASETS:
                    P = paper_json(bb, ds) or {}
                    v = P.get("calibrators", {}).get(full) if key is None else P.get(sec, {}).get(key)
                    vals.append(v["auroc"]["mean"] if v else None)
                cells.append(fmt3(np.mean(vals)) if all(v is not None for v in vals) else "--")
        shade = r"\rowcolor{gray!20} " if key is None else ""
        lines.append(f"{shade}{label} & " + " & ".join(cells) + r" \\")
    return wrap_table(lines, cols, "Stage-1 ablations for all backbones: AUROC averaged over the six datasets, with the full calibrator (\\m) and without any judge input (\\mA). Every variant keeps the same R-Log. Llama uses 10 seeds; for Qwen3 and Ministral, the head swaps and PCS ablations use 5 seeds and the architecture ablations are not run.",
                      "tab:ablation_all", star=True, sep="5pt")


def internalization_table():
    """How well head step scores / predicted verdicts track the judge on test (mean over 10 seeds)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from cuq.io import index_by_uid, read_jsonl
    from cuq.registry import HEAD_SEEDS, judge_path, pred_path
    rows = [("No PCS", "chainuq2bd-core"), ("PCS (ours)", "chainuq2pcs")]
    cols = "ll" + "c" * len(DATASETS)
    lines = [r"\toprule", " & ".join([r"\textbf{Backbone}", r"\textbf{Head}"] + [n.replace(" (ID)", "") for _, n in DATASETS]) + r" \\", r"\midrule"]
    stats = {}
    for bb, bname in BACKBONES:
        for label, head in rows:
            cells = []
            for ds, _ in DATASETS:
                steps = index_by_uid(read_jsonl(judge_path("gptoss", "steps", bb, ds, "test")))
                fb = index_by_uid(read_jsonl(judge_path("gptoss", "final_blind", bb, ds, "test")))
                aucs, rhos = [], []
                for sd in HEAD_SEEDS:
                    pth = pred_path(head, bb, sd, ds, "test")
                    if not pth.exists():
                        continue
                    P = index_by_uid(read_jsonl(pth))
                    a, b = [], []
                    for u, p in P.items():
                        lab = steps.get(u, {}).get("value")
                        if isinstance(lab, list) and p.get("step_logits") and len(lab) == len(p["step_logits"]):
                            a += lab
                            b += p["step_logits"]
                    if len(set(a)) > 1:
                        aucs.append(roc_auc_score(a, b))
                    vh = [(p["verdict_hat"], fb[u]["p_verbal"]) for u, p in P.items()
                          if "verdict_hat" in p and u in fb and fb[u].get("p_verbal") is not None]
                    if vh:
                        rhos.append(spearmanr([x for x, _ in vh], [y for _, y in vh]).correlation)
                stats[(bb, head, ds)] = (float(np.mean(aucs)) if aucs else None, float(np.mean(rhos)) if rhos else None)
                auc, rho = stats[(bb, head, ds)]
                cells.append("--" if auc is None else (f"{auc:.3f}" if rho is None else f"{auc:.3f} / {rho:.2f}"))
            lines.append(f"{bname.split('-')[0] if label == rows[0][0] else ''} & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    tex = wrap_table(lines, cols, "Consistency internalization on the test splits (mean over 10 seeds): AUROC of the head's step scores against the judge's answer-blind step labels, and, for PCS, Spearman correlation of the predicted verdict $\\hat v(r)$ with the judge's verdict. No judge is queried by either head at inference.",
                     "tab:internalize", star=True, sep="4pt")
    return tex, stats


def test2_table():
    cols, lines = "ll ccc ccc", [r"\toprule"]
    lines.append(r" & & \multicolumn{3}{c}{\textbf{With external verifier}} & \multicolumn{3}{c}{\textbf{Judge-free}} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r"\textbf{Backbone} & \textbf{Dataset} & \m & Best baseline & & \mA & Best baseline & \\")
    lines.append(r"\midrule")
    tier_of = {k: t for _, ms in GROUPS for k, t, _ in ms}
    disp = {k: n for _, ms in GROUPS for k, _, n in ms}
    for bb, bname in BACKBONES:
        res = load(bb, "test2")["datasets"]
        for i, (ds, dname) in enumerate([d for d in DATASETS if d[0] in res]):
            R = res[ds]
            def best(pool):
                c = {k: value(R, k, tier_of[k], "auroc") for k in pool}
                c = {k: v for k, v in c.items() if v is not None}
                k = max(c, key=c.get)
                return disp[k], c[k]
            nb, vb = best(BASELINES)
            na, va = best(TIER_A_BASELINES)
            o, oa = value(R, OURS, "tier_b", "auroc"), value(R, OURS_A, "tier_a", "auroc")
            b = lambda x, y: rf"\textbf{{{x:.3f}}}" if x >= y else f"{x:.3f}"
            lines.append(f"{bname.split('-')[0] if i == 0 else ''} & {dname.replace(' (ID)', '')} & {b(o, vb)} & {vb:.3f} & {{\\scriptsize {nb}}} & "
                         f"{b(oa, va)} & {va:.3f} & {{\\scriptsize {na}}} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    return wrap_table(lines, cols, "AUROC on the fresh held-out Test-2 questions (single evaluation of the frozen pipeline). \\textbf{Bold}: \\m or \\mA is at least as good as the strongest baseline of its setting.",
                      "tab:test2", star=True, sep="4pt")


OOD_NAMES = {"musique": "MuSiQue", "strategyqa": "StrategyQA", "babi": "bAbI", "gsm8k": "GSM8K", "math500": "MATH-500"}
MARKERS = {"llama": "D", "qwen8b": "o", "ministral": "s"}
SHORT = {"llama": "Llama", "qwen8b": "Qwen", "ministral": "Ministral"}
COLORS = {"musique": "#4C78A8", "strategyqa": "#F58518", "babi": "#54A24B", "gsm8k": "#B279A2", "math500": "#E45756"}


def figure_points():
    pts = []
    for bb, _ in BACKBONES:
        for ds in OOD_NAMES:
            P = paper_json(bb, ds)
            if not P:
                continue
            r, i = P["calibrators"]["rlog"], P["calibrators"]["iso"]
            pts.append({"bb": bb, "ds": ds, "dprauc": r["prauc"]["mean"] - i["prauc"]["mean"],
                        "dcalq": i["ece"]["mean"] - r["ece"]["mean"], "hard": 1 - i["prauc"]["mean"]})
    return pts


def draw_figures(fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 17, "axes.linewidth": 1.6,
                         "xtick.direction": "in", "ytick.direction": "in"})
    pts = figure_points()
    legend = ([Line2D([], [], marker=MARKERS[b], color="w", markerfacecolor="white", markeredgecolor="black",
                      markersize=10, label=SHORT[b]) for b, _ in BACKBONES] +
              [Line2D([], [], marker="o", color="w", markerfacecolor=COLORS[d], markeredgecolor="black",
                      markersize=10, label=n) for d, n in OOD_NAMES.items()])

    def base(ax, loc="best", anchor=None):
        for p_ in pts:
            ax.scatter(p_["x"], p_["y"], marker=MARKERS[p_["bb"]], s=170, c=COLORS[p_["ds"]], edgecolors="black",
                       linewidths=0.6, zorder=3)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(handles=legend, ncol=2, fontsize=11.5, loc=loc, bbox_to_anchor=anchor, framealpha=0.9, handletextpad=0.2,
                  columnspacing=0.6, borderpad=0.4)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p_ in pts:
        p_["x"], p_["y"] = p_["dprauc"], p_["dcalq"]
    base(ax)
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    ax.axvline(0, color="black", lw=0.9, zorder=1)
    ax.set_xlabel(r"Ranking gain ($\Delta$PRAUC)")
    ax.set_ylabel(r"Calibration gain ($\Delta$CalQ)")
    fig.tight_layout()
    fig.savefig(fig_dir / "advantage_map.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p_ in pts:
        p_["x"], p_["y"] = p_["hard"], p_["dprauc"]
    base(ax, loc="upper left", anchor=(0.0, 0.9))
    x, y = np.array([p_["x"] for p_ in pts]), np.array([p_["y"] for p_ in pts])
    r = float(np.corrcoef(x, y)[0, 1])
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, k * xs + b, color="#B22222", lw=2.4, zorder=2)
    ax.text(0.04, 0.95, f"r (gain, hardness) = {r:.3f}", transform=ax.transAxes, color="#B22222", va="top", fontsize=16)
    ax.set_xlabel("OOD hardness")
    ax.set_ylabel(r"Ranking gain ($\Delta$PRAUC)")
    fig.tight_layout()
    fig.savefig(fig_dir / "frontier.pdf")
    plt.close(fig)
    return {"pearson_r": r, "n_points": len(pts), "pts": pts}


def numbers_tex(summary, extra):
    B, A = summary["test"]["B"], summary["test"]["A"]
    T2B, T2A = summary["test2"]["B"], summary["test2"]["A"]
    pct = lambda v: f"{100 * v:.1f}\\%"
    out = {"AUROCGAIN": pct(B["mean_rel_auroc_gain"]), "AUROCGAINMAX": pct(B["max_rel_auroc_gain"]),
           "AUROCGAINA": pct(A["mean_rel_auroc_gain"])}
    for tier, d in (("B", B), ("A", A), ("TwoB", T2B), ("TwoA", T2A)):
        for m, (w, n) in d["per_metric"].items():
            out[f"Win{tier}{m.replace('90', '')}"] = str(w)
        out[f"Win{tier}total"] = f"{d['total'][0]}/{d['total'][1]}"
    out.update(extra)
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in out.items()) + "\n"


CASE_PLAN = [("hotpotqa", 1, "correct response raised"), ("hotpotqa", 0, "plausible but wrong conclusion lowered"),
             ("musique", 0, "multi-hop composition drifts"), ("babi", 1, "clean symbolic chain reinforced"),
             ("gsm8k", 0, "arithmetic slip lowered"), ("math500", 1, "correct derivation raised")]


def _esc(t):
    t = str(t)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        t = t.replace(a, b)
    return t


def cases_tex(bb="llama"):
    from cuq.io import index_by_uid, read_jsonl
    from cuq.registry import gen_path
    cards, used = [], []
    for k, (ds, want_y, title) in enumerate(CASE_PLAN):
        path = results_dir() / "paper" / bb / f"{ds}_test_cases.json"
        if not path.exists():
            continue
        cases = json.loads(path.read_text())
        gens = index_by_uid(read_jsonl(gen_path(bb, ds, "test")))
        def ok(c):
            g = gens.get(c["uid"])
            return (g and c["y"] == want_y and c["step_labels"] and 2 <= len(g["steps"]) <= 5
                    and len(g["question"]) < 220 and len(g["gen_text"]) < 700 and len(str(g["gold"])) <= 40
                    and ((c["conf"] - c["s"] > 0.3) if want_y == 1 else (c["s"] - c["conf"] > 0.3)))
        pool = [c for c in cases if ok(c)]
        if not pool:
            continue
        c = max(pool, key=lambda c: abs(c["conf"] - c["s"]))
        g = gens[c["uid"]]
        good = want_y == 1
        col = "blue!60!black" if good else "red!65!black"
        bg = "blue!4" if good else "red!4"
        steps = " ".join(rf"\textbf{{Step {i + 1}:}} {_esc(st['text'])}." for i, st in enumerate(g["steps"]))
        tags = " ".join(rf"\scoretag{{{'green!18' if v == 1 else 'red!18'}}}{{S{i + 1}: {v}}}"
                        for i, v in enumerate(c["step_labels"]))
        verdict = "--" if c["verdict"] is None else f"{c['verdict']:.2f}"
        name = dict(DATASETS)[ds].replace(" (ID)", "")
        cards.append(rf"""\noindent\makebox[\textwidth][c]{{\fcolorbox{{{col}}}{{{bg}}}{{
\begin{{minipage}}[t]{{\dimexpr\appendixcardwidth-2\fboxsep-2\fboxrule\relax}}
\small
\renewcommand{{\arraystretch}}{{1.22}}
\cardtitle{{{col}}}{{Example {chr(65 + len(cards))} ({name}): {title}}}
\begin{{tabular}}{{@{{}}p{{0.18\linewidth}}p{{0.76\linewidth}}@{{}}}}
\textbf{{Question}} &
{_esc(g['question'])}\\[4pt]
\textbf{{LLM output}} &
\parbox[t]{{0.92\linewidth}}{{\footnotesize\raggedright
Reasoning: {steps} Conclusion: \textcolor{{{'green!50!black' if good else 'red!70!black'}}}{{{_esc(g['conclusion']['text'])}}}. \quad Reference: {_esc(g['gold'])}}}\\[4pt]
\textbf{{Judge / scores}} &
\parbox[t]{{0.92\linewidth}}{{%
\scoretag[7.2em]{{{'green!18' if good else 'red!18'}}}{{{'correct (1)' if good else 'incorrect (0)'}}} \quad {tags}\\[4pt]
Chain verdict $v(r)$: {verdict} \quad Alignment-aware UQ $\boldsymbol{{s}}$: {c['s']:.3f} \quad Calibrated confidence: \scoretag{{{'green!18' if good else 'red!18'}}}{{{c['conf']:.3f}}}}}
\end{{tabular}}
\end{{minipage}}}}}}
""")
        used.append(c["uid"])
    intro = (r"\section{Case Studies}\label{sec:app_case_studies}" "\n\n"
             r"We show illustrative test examples from the Llama run (seed 2026), one per dataset group, selected among "
             r"short responses where the calibrated confidence moves far from the head score in the direction of the "
             r"correctness label. Step labels and the chain verdict come from the judge without the reference answer; "
             r"the reference answer is shown only for the reader." "\n\n")
    return intro + "\n\\vspace{8pt}\n".join(cards), used


def fmt_ms(v):
    """Latency in ms with three significant digits; sub-microsecond values as an upper bound."""
    return "$<$0.001" if v < 0.001 else (f"{v:.3f}" if v < 1 else f"{v:.1f}")


COTUQ_SECONDS, COTUQ_RESPONSES = 213.0, 19906  # llama CoT-UQ keyword call over all val/test responses (vLLM log)


def efficiency_tables(bb="llama", ds="hotpotqa"):
    bench = json.loads((results_dir() / "paper" / f"bench_{bb}_{ds}.json").read_text())
    costs = {r["stage"]: r for r in json.loads((results_dir() / "costs.json").read_text()) if r["backbone"] == bb}
    ms = lambda stage: 1000 * costs[stage]["sec_per_sample"]
    H, Sc, Cb = bench["heads"], bench["scores"], bench["calibrators"]
    rlog_ms = Cb["rlog"]["ms_per_sample"]
    judge_steps, judge_final = ms("judge[gptoss/steps]"), ms("judge[gptoss/final_blind]")
    sampling = ms("gen[step/samples]")
    rows = [("MCP", "--", Sc["mcp"]["ms_per_sample"], None), ("Perplexity", "--", Sc["ppl"]["ms_per_sample"], None),
            ("MTE", "--", Sc["mte"]["ms_per_sample"], None), ("CCP", "NLI", Sc["ccp"]["ms_per_sample"], Sc["ccp"]["peak_gpu_gb"]),
            ("CoT-UQ", "1 backbone call", 1000 * COTUQ_SECONDS / COTUQ_RESPONSES, None),
            ("Semantic Entropy", "10 samples + NLI", sampling + Sc["se_selfcheck_nli"]["ms_per_sample"], None),
            ("SelfCheckGPT", "10 samples + NLI", sampling + Sc["se_selfcheck_nli"]["ms_per_sample"], None),
            ("SAPLMA", "--", H["saplma"]["ms_per_sample"], H["saplma"]["peak_gpu_gb"]),
            ("Factoscope", "--", H["factoscope"]["ms_per_sample"], H["factoscope"]["peak_gpu_gb"]),
            ("Lookback Lens", "--", H["lookback"]["ms_per_sample"], None),
            ("UHead", "--", H["uhead"]["ms_per_sample"], H["uhead"]["peak_gpu_gb"]),
            ("MeanPool-MLP", "--", H["meanpool-mlp"]["ms_per_sample"], H["meanpool-mlp"]["peak_gpu_gb"]),
            ("AttnPool-MLP", "--", H["attnpool-mlp"]["ms_per_sample"], H["attnpool-mlp"]["peak_gpu_gb"]),
            (r"\textbf{\mA}", "--", H[FINAL_HEAD].get("ms_per_sample_full", H[FINAL_HEAD]["ms_per_sample"]) + rlog_ms, H[FINAL_HEAD]["peak_gpu_gb"]),
            ("Judge-Step", "1 judge call", judge_steps, None), ("Judge-Final", "1 judge call", judge_final, None),
            ("Reward Model", "1 RM forward", Sc["rm"]["ms_per_sample"], Sc["rm"]["peak_gpu_gb"]),
            (r"\textbf{\m}", "2 judge calls", H[FINAL_HEAD].get("ms_per_sample_full", H[FINAL_HEAD]["ms_per_sample"]) + rlog_ms + judge_steps + judge_final,
             H[FINAL_HEAD]["peak_gpu_gb"])]
    lines = [r"\toprule", r"\textbf{Method} & \textbf{Extra model calls} & \textbf{Latency (ms)} $\downarrow$ & \textbf{Head GPU mem (GB)} $\downarrow$ \\", r"\midrule"]
    for name, calls, lat, mem in rows:
        shade = r"\rowcolor{gray!20} " if "\\m" in name else ""
        lines.append(f"{shade}{name} & {calls} & {fmt_ms(lat)} & {'--' if mem is None else f'{mem:.2f}'} \\\\")
    common = (ms("gen[step]"), ms("features[step]"))
    t1 = wrap_table(lines, "l l r r", (
        f"Post-generation cost per response on {bench['gpu'].replace('NVIDIA ', '')} (Llama-3.1-8B-Instruct, HotpotQA test). "
        f"LLM, judge, and NLI calls are amortized under batched serving; heads run in batches of 64 on cached features. "
        f"Not included and shared by all methods: greedy generation ({common[0]:.0f} ms) and, for white-box features, "
        f"one teacher-forced forward pass ({common[1]:.0f} ms)."), "tab:efficiency", sep="3pt")
    lines = [r"\toprule", r"\textbf{Calibrator} & \textbf{Fit on validation (s)} $\downarrow$ & \textbf{Predict (ms/sample)} $\downarrow$ \\", r"\midrule"]
    for key, name in CALIBRATORS:
        shade = r"\rowcolor{gray!20} " if key == "rlog" else ""
        lines.append(f"{shade}{name} & {Cb[key]['fit_s']:.2f} & {fmt_ms(Cb[key]['ms_per_sample'])} \\\\")
    t2 = wrap_table(lines, "l r r", (f"Cost of the post-hoc calibrators on the \\m head (Llama, HotpotQA; {Cb['n_val']} validation and {Cb['n_test']} test responses, one CPU core). "
                                     "BWH and R-Log include building the evidence vector $u(r)$."), "tab:efficiency_post", sep="4pt")
    summ = {"chainuq_ms": rows[-1][2], "chainuqA_ms": rows[13][2], "se_ms": rows[5][2], "judge_final_ms": judge_final,
            "head_ms": H[FINAL_HEAD].get("ms_per_sample_full", H[FINAL_HEAD]["ms_per_sample"]), "rlog_ms": rlog_ms, "rlog_fit_s": Cb["rlog"]["fit_s"],
            "gen_ms": common[0], "feat_ms": common[1], "uhead_gb": H["uhead"]["peak_gpu_gb"], "ours_gb": H[FINAL_HEAD]["peak_gpu_gb"]}
    return t1, t2, summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", default=str(Path(__file__).resolve().parents[2] / "latex"))
    args = p.parse_args()
    tdir, fdir = Path(args.latex) / "table", Path(args.latex) / "fig"
    disc, unc = ["acc", "auroc", "prauc"], ["ece", "brier", "risk90"]
    names = dict(BACKBONES)
    (tdir / "result.tex").write_text(method_table(
        "llama", disc, f"Discrimination results of {names['llama']} on six datasets (mean over 10 seeds). Groups: "
        r"single-response unsupervised; extra backbone calls; supervised probes; \mA; external verifiers; \m. "
        r"\textbf{Bold}: best; \underline{underline}: second best. \mA uses no external model at inference.",
        "tab:result"))
    app = [method_table(bb, disc, f"Discrimination results of {name} on six datasets.", f"tab:result_{i}")
           for i, (bb, name) in enumerate(BACKBONES[1:], start=2)]
    app += [method_table(bb, unc, f"Calibration and selective-risk results of all methods for {name} "
                                  "(baselines calibrated with isotonic regression on validation).", f"tab:uq_{i}")
            for i, (bb, name) in enumerate(BACKBONES, start=1)]
    (tdir / "app_results.tex").write_text("\n".join(app))
    summary = {"test": wins("test"), "test2": wins("test2")}
    extra = {}
    have = {bb: all(paper_json(bb, ds) for ds, _ in DATASETS) for bb, _ in BACKBONES}
    if have["llama"]:
        (tdir / "calibration.tex").write_text(calibrators_for("llama"))
        abl, _ = ablation_table()
        (tdir / "ablation.tex").write_text(abl)
        (tdir / "ablation_all.tex").write_text(ablation_all_table())
    if have["qwen8b"] and have["ministral"]:
        (tdir / "app_calibration.tex").write_text(calibrators_for("qwen8b") + "\n" + calibrators_for("ministral"))
    if all(have.values()):
        js, means = judge_swap_table()
        (tdir / "judge_swap.tex").write_text(js)
        fig = draw_figures(fdir)
        extra["PearsonR"] = f"{fig['pearson_r']:.3f}"
        red = []
        for bb, _ in BACKBONES:
            for ds, _ in DATASETS:
                c = paper_json(bb, ds)["calibrators"]
                red.append((c["iso"]["ece"]["mean"] - c["rlog"]["ece"]["mean"]) / c["iso"]["ece"]["mean"])
        extra["ECEGAIN"] = f"{100 * max(red):.1f}\\%"
        extra["ECEWinPairs"] = str(sum(r > 0 for r in red))
        extra["ECEMedianGain"] = f"{100 * float(np.median(red)):.0f}\\%"
        best = 0
        for ds, _ in DATASETS:
            c = paper_json("llama", ds)["calibrators"]
            for mm in ("ece", "brier", "risk90"):
                v = {k: round(c[k][mm]["mean"], 3) for k, _ in CALIBRATORS}
                best += v["rlog"] <= min(v.values())
        extra["CalBestLlama"] = str(best)
        cm = paper_json("llama", "math500")["calibrators"]
        extra["MathBrierRLog"], extra["MathBrierISO"] = f"{cm['rlog']['brier']['mean']:.3f}", f"{cm['iso']['brier']['mean']:.3f}"
        extra["MathRiskRLog"], extra["MathRiskISO"] = f"{cm['rlog']['risk90']['mean']:.3f}", f"{cm['iso']['risk90']['mean']:.3f}"
        pts = fig["pts"]
        extra["CalQPos"] = str(sum(p_["dcalq"] > 0 for p_ in pts))
        extra["PRAUCPos"] = str(sum(p_["dprauc"] > 0 for p_ in pts))
        extra["NOODPairs"] = str(len(pts))
        from collections import Counter
        ch = Counter()
        for bb, _ in BACKBONES:
            for ds, _ in DATASETS:
                ch.update(paper_json(bb, ds).get("choice", {}))
        extra["CapTotal"] = str(sum(ch.values()))
        extra["CapRLogPlatt"] = str(ch.get("rlogcf_c0.05_platt", 0))
        extra["CapGBM"] = str(ch.get("gbmcf_iso", 0))
        extra["CapLinear"] = str(ch.get("lr_c0.01", 0))
        extra["CapAnchor"] = str(ch.get("iso_s", 0))
        extra["CapOther"] = str(sum(ch.values()) - ch.get("rlogcf_c0.05_platt", 0) - ch.get("gbmcf_iso", 0)
                                - ch.get("lr_c0.01", 0) - ch.get("iso_s", 0))
        extra["StepGain"] = f"{means['gptoss'] - means['none']:.3f}"
        summary["ece_rel_reduction_vs_iso"] = red
        summary["judge_swap_means"] = means
        summary["figure"] = fig
    if FINAL_HEAD == "chainuq2pcs":
        it, it_stats = internalization_table()
        (tdir / "internalize.tex").write_text(it)
        gains = [it_stats[(bb, "chainuq2pcs", ds)][0] - it_stats[(bb, "chainuq2bd-core", ds)][0]
                 for bb, _ in BACKBONES for ds, _ in DATASETS
                 if it_stats[(bb, "chainuq2pcs", ds)][0] is not None and it_stats[(bb, "chainuq2bd-core", ds)][0] is not None]
        extra["StepAUROCGain"] = f"{np.mean(gains):.3f}"
        extra["StepAUROCPCS"] = f"{np.mean([it_stats[(bb, 'chainuq2pcs', ds)][0] for bb, _ in BACKBONES for ds, _ in DATASETS]):.3f}"
        extra["StepAUROCCore"] = f"{np.mean([it_stats[(bb, 'chainuq2bd-core', ds)][0] for bb, _ in BACKBONES for ds, _ in DATASETS]):.3f}"
        extra["VerdictRho"] = f"{np.mean([it_stats[(bb, 'chainuq2pcs', ds)][1] for bb, _ in BACKBONES for ds, _ in DATASETS]):.2f}"
    ag, ag_stats = agreement_table()
    (tdir / "agreement.tex").write_text(ag)
    (tdir / "test2.tex").write_text(test2_table())
    t2b, t2a, tb, ta, gaps, weak = [], [], [], [], [], 0
    tier_of = {k: t for _, ms in GROUPS for k, t, _ in ms}
    for bb, _ in BACKBONES:
        r2, r1 = load(bb, "test2")["datasets"], load(bb)["datasets"]
        for ds, R in r2.items():
            t2b.append(value(R, OURS, "tier_b", "auroc")); t2a.append(value(R, OURS_A, "tier_a", "auroc"))
            tb.append(value(r1[ds], OURS, "tier_b", "auroc")); ta.append(value(r1[ds], OURS_A, "tier_a", "auroc"))
            cand = {k: value(R, k, tier_of[k], "ece") for k in BASELINES}
            cand = {k: v for k, v in cand.items() if v is not None}
            kbest = min(cand, key=cand.get)
            gaps.append(value(R, OURS, "tier_b", "ece") - cand[kbest])
            weak += (value(R, kbest, tier_of[kbest], "auroc") or 1) < 0.75
    rl = load("llama")["datasets"]
    for ds, key in (("babi", "Babi"), ("musique", "Mus")):
        R = rl[ds]
        cand = {k: value(R, k, tier_of[k], "auroc") for k in BASELINES}
        cand = {k: v for k, v in cand.items() if v is not None}
        extra[f"Llama{key}Ours"] = f"{value(R, OURS, 'tier_b', 'auroc'):.3f}"
        extra[f"Llama{key}Best"] = f"{max(cand.values()):.3f}"
    jf_gap = [value(load(bb)["datasets"][ds], OURS, "tier_b", "auroc") - value(load(bb)["datasets"][ds], "judge_final", "tier_b", "auroc")
              for bb, _ in BACKBONES for ds, _ in DATASETS]
    extra["JudgeFinalGap"] = f"{np.mean(jf_gap):.3f}"
    extra["JudgeFinalWins"] = str(sum(g >= 0 for g in jf_gap))
    extra.update(TwoMeanB=f"{np.mean(t2b):.3f}", TwoMeanA=f"{np.mean(t2a):.3f}", TestSameMeanB=f"{np.mean(tb):.3f}",
                 TestSameMeanA=f"{np.mean(ta):.3f}", TwoECEMaxGap=f"{max(gaps):.3f}", TwoECEWeak=str(weak),
                 TwoPairs=str(len(t2b)))
    eces = [load(bb)["datasets"][ds]["tier_b"][OURS]["ece"]["mean"] for bb, _ in BACKBONES for ds, _ in DATASETS]
    extra["MeanECE"] = f"{np.mean(eces):.3f}"
    extra.update({"LabelKappaMin": f"{ag_stats['label_kappa_min']:.2f}", "LabelKappaMax": f"{ag_stats['label_kappa_max']:.2f}",
                  "StepKappaMax": f"{ag_stats['step_kappa_max']:.2f}"})
    if (results_dir() / "paper" / "bench_llama_hotpotqa.json").exists():
        t1, t2, eff = efficiency_tables()
        (tdir / "efficiency.tex").write_text(t1)
        (tdir / "efficiency_post.tex").write_text(t2)
        summary["efficiency"] = eff
        extra.update(EffHeadMs=fmt_ms(eff["head_ms"]), EffRLogMs=fmt_ms(eff["rlog_ms"]), EffChainUQMs=f"{eff['chainuq_ms']:.0f}",
                     EffChainUQAMs=f"{eff['chainuqA_ms']:.0f}", EffSEMs=f"{eff['se_ms']:.0f}", EffGenMs=f"{eff['gen_ms']:.0f}",
                     EffFeatMs=f"{eff['feat_ms']:.0f}")
    if have["llama"]:
        cases, used = cases_tex()
        (tdir / "cases.tex").write_text(cases)
        summary["case_uids"] = used
    (tdir / "numbers.tex").write_text(numbers_tex(summary, extra))
    (results_dir() / "paper").mkdir(parents=True, exist_ok=True)
    (results_dir() / "paper" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print((tdir / "numbers.tex").read_text())


if __name__ == "__main__":
    main()
