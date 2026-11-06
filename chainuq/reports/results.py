"""Main-comparison, calibrator, significance, Test-2 and efficiency tables of the paper.

Inputs (under ``results/``): ``<backbone>/eval.json``, ``eval_test2.json`` (``pipeline.evaluate``) and
``eval_extra*.json`` (``analysis.fusion``); ``paper/<backbone>/<dataset>_test.json`` (``analysis.calibrators``) and
``_signif.json`` (``analysis.significance``); ``paper/bench_llama_hotpotqa.json`` (``analysis.benchmark``) and
``costs.json`` (``analysis.costs``). Writes ``table/{result,app_results,app_calibration,significance,test2,
efficiency,efficiency_post}.tex`` under ``CHAINUQ_PAPER_DIR``.

    python -m chainuq.reports.results
"""

from __future__ import annotations

import json

import numpy as np

from chainuq.artifacts import results_dir
from chainuq.config import PAPER_DIR
from chainuq.registry import BASELINE_GROUPS, FINAL_HEAD, OURS, OURS_A
from chainuq.reports.tables import readable_table

BACKBONES = [("llama", "Llama-3.1-8B-Instruct"), ("qwen8b", "Qwen3-8B"), ("ministral", "Ministral-8B-Instruct-2410")]
BB_NAME = {"llama": "Llama-3.1-8B", "qwen8b": "Qwen3-8B", "ministral": "Ministral-8B"}
DATASETS = [("hotpotqa", "HotpotQA (ID)"), ("musique", "MuSiQue"), ("strategyqa", "StrategyQA"), ("babi", "bAbI"), ("gsm8k", "GSM8K"), ("math500", "MATH-500")]
LOWER = {"ece", "brier", "risk90", "aurc"}
MNAME = {"acc": "Acc", "auroc": "AUROC", "prauc": "AP", "ece": "ECE", "brier": "Brier", "risk90": "Risk@90", "aurc": "AURC"}
# (group label, [(key, tier, display name)]): rows of the main comparison, ChainUQ-A after the judge-free groups.
FUSION_GROUP = ("Probe + judge fusion", [("fusion-saplma", "tier_b", "SAPLMA + Judge"), ("fusion-uhead", "tier_b", "UHead + Judge")])
GROUPS = (
    BASELINE_GROUPS[:3]
    + [("OURS_A", [(OURS_A, "tier_a", r"\textbf{\m}")])]
    + BASELINE_GROUPS[3:]
    + [FUSION_GROUP, ("OURS", [(OURS, "tier_b", r"\textbf{\mJ}")])]
)
BASELINES = [k for g, ms in GROUPS for k, _, _ in ms if not g.startswith("OURS")]
TIER_A_BASELINES = [k for g, ms in GROUPS if g in ("Single-response, unsupervised", "Extra backbone calls", "Supervised probes") for k, _, _ in ms]


def load(bb, split="test"):
    R = json.loads((results_dir() / bb / ("eval.json" if split == "test" else f"eval_{split}.json")).read_text())
    # probe + judge fusion baselines (analysis.fusion)
    extra = results_dir() / bb / ("eval_extra.json" if split == "test" else f"eval_extra_{split}.json")
    if extra.exists():
        for ds, cell in json.loads(extra.read_text())["datasets"].items():
            for tier, methods in cell.items():
                R["datasets"].setdefault(ds, {}).setdefault(tier, {}).update(methods)
    return R


def paper_json(bb, ds):
    p = results_dir() / "paper" / bb / f"{ds}_test.json"
    return json.loads(p.read_text()) if p.exists() else None


def value(R, key, tier, metric):
    v = R[tier].get(key, {}).get(metric)
    return None if v is None else v["mean"]


_AURC = {}


def aurc_value(key, bb, ds):
    """Test AURC averaged over seeds, from the per-response predictions (analysis.responses)."""
    if (key, bb, ds) not in _AURC:
        from chainuq.analysis import variants as D
        from chainuq.analysis.fusion import FUSION
        from chainuq.analysis.responses import aurc, baseline_scores
        from chainuq.analysis.responses import load as load_preds

        src = {OURS: D.CHAINUQ, OURS_A: D.CHAINUQ_A, **FUSION}
        z = load_preds(src.get(key, D.CHAINUQ), bb, ds)
        S = z["p"] if key in src else baseline_scores(key, bb, ds, [str(u) for u in z["uids"]])
        _AURC[(key, bb, ds)] = None if S is None else float(np.mean([aurc(z["y"], s) for s in S]))
    return _AURC[(key, bb, ds)]


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


def table_groups(aggregate=False):
    if aggregate:
        return [
            ("id", "HotpotQA (ID)", ["hotpotqa"]),
            ("ood", "OOD", [ds for ds, _ in DATASETS if ds != "hotpotqa"]),
            ("macro", "Macro", [ds for ds, _ in DATASETS]),
        ]
    return [(ds, name, [ds]) for ds, name in DATASETS]


def complete_mean(values):
    """Equal dataset weights; a missing cell cannot silently change the cohort."""
    return float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None


def method_table(bb, metrics, caption, label, star=True, aggregate=False):
    res = load(bb)["datasets"]
    rows = [(g, k, t, name) for g, ms in GROUPS for k, t, name in ms]
    groups = table_groups(aggregate)
    # two inference-cost blocks, ranked separately: no external model (up to ChainUQ-A), and one judge/verifier call
    cut = next(i for i, r in enumerate(rows) if r[0] == "OURS_A") + 1
    blocks = [("No judge or verifier call", rows[:cut]), ("One judge or verifier call per response", rows[cut:])]
    cells = {}
    for _, brows in blocks:
        for group, _, members in groups:
            for m in metrics:
                vals = [complete_mean([value(res[ds], k, t, m) if ds in res else None for ds in members]) for _, k, t, _ in brows]
                for (_, k, _, _), s in zip(brows, rank_marks(vals, m)):
                    cells[(k, group, m)] = s
    cols, lines = header(metrics, [(key, name) for key, name, _ in groups])
    ncol = 1 + len(groups) * len(metrics)
    prev = None
    for g, k, t, name in rows:
        start = next((bname for bname, brows in blocks if brows[0][1] == k), None)
        if start:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{start}}}}} \\")  # 4 columns: also fits the split appendix tables
        elif g != prev and prev is not None:
            lines.append(r"\cmidrule(lr){1-%d}" % ncol)
        prev = g
        body = " & ".join(cells[(k, group, m)] for group, _, _ in groups for m in metrics)
        shade = r"\rowcolor{gray!20} " if g.startswith("OURS") else ""
        lines.append(f"{shade}{name}{'$^*$' if k in SUBSET else ''} & {body} \\\\")
    lines.append(r"\bottomrule")
    env = "table*" if star else "table"
    return readable_table(
        "\n".join(
            [
                rf"\begin{{{env}}}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{3pt}",
                rf"\begin{{tabular}}{{{cols}}}",
                *lines,
                r"\end{tabular}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                rf"\end{{{env}}}",
                "",
            ]
        )
    )


SUBSET = {"se", "selfcheck"}  # evaluated on the first 1,000 test questions of each dataset


def backbone_table(metrics, caption, label):
    """Main comparison: macro over the six datasets per backbone, and the mean over backbones."""
    res = {bb: load(bb)["datasets"] for bb, _ in BACKBONES}
    rows = [(g, k, t, name) for g, ms in GROUPS for k, t, name in ms]
    cut = next(i for i, r in enumerate(rows) if r[0] == "OURS_A") + 1
    blocks = [("No judge or verifier call", rows[:cut]), ("One judge or verifier call per response", rows[cut:])]
    groups = [(bb, BB_NAME[bb]) for bb, _ in BACKBONES] + [("mean", "Mean")]

    def cell(k, t, m, bb):
        if bb == "mean":
            per = [cell(k, t, m, b) for b, _ in BACKBONES]
            return complete_mean(per) if all(v is not None for v in per) else None
        if m == "aurc":
            return complete_mean([aurc_value(k, bb, ds) for ds, _ in DATASETS])
        R = res[bb]
        return complete_mean([value(R[ds], k, t, m) if ds in R else None for ds, _ in DATASETS])

    cells = {}
    for _, brows in blocks:
        for bb, _ in groups:
            for m in metrics:
                vals = [cell(k, t, m, bb) for _, k, t, _ in brows]
                for (_, k, _, _), s in zip(brows, rank_marks(vals, m)):
                    cells[(k, bb, m)] = s
    cols, lines = header(metrics, groups)
    ncol = 1 + len(groups) * len(metrics)
    prev = None
    for g, k, t, name in rows:
        start = next((bname for bname, brows in blocks if brows[0][1] == k), None)
        if start:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{start}}}}} \\")
        elif g != prev and prev is not None:
            lines.append(r"\cmidrule(lr){1-%d}" % ncol)
        prev = g
        body = " & ".join(cells[(k, bb, m)] for bb, _ in groups for m in metrics)
        shade = r"\rowcolor{gray!20} " if g.startswith("OURS") else ""
        lines.append(f"{shade}{name}{'$^*$' if k in SUBSET else ''} & {body} \\\\")
    lines.append(r"\bottomrule")
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\begin{{tabular}}{{{cols}}}",
            *lines,
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )


CALIBRATORS = [("ps", "PS"), ("ts", "TS"), ("iso", "ISO"), ("bwh", "BWH"), ("rlog", r"\textbf{Ours (\mJ)}")]


def calibrator_table(bb, caption, label):
    metrics = ["ece", "brier", "risk90"]
    groups = table_groups()
    results = {ds: paper_json(bb, ds) for ds, _ in DATASETS}
    cells = {}
    for group, _, members in groups:
        for m in metrics:
            vals = [
                complete_mean([results[ds]["calibrators"][k][m]["mean"] if results[ds] and k in results[ds].get("calibrators", {}) else None for ds in members])
                for k, _ in CALIBRATORS
            ]
            for (k, _), s in zip(CALIBRATORS, rank_marks(vals, m)):
                cells[(k, group, m)] = s
    cols, lines = header(metrics, [(key, name) for key, name, _ in groups])
    for k, name in CALIBRATORS:
        shade = r"\rowcolor{gray!20} " if k == "rlog" else ""
        lines.append(f"{shade}{name} & " + " & ".join(cells[(k, group, m)] for group, _, _ in groups for m in metrics) + r" \\")
    lines.append(r"\bottomrule")
    return readable_table(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{3pt}",
                rf"\begin{{tabular}}{{{cols}}}",
                *lines,
                r"\end{tabular}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table*}",
                "",
            ]
        )
    )


def wrap_table(lines, cols, caption, label, star=False, sep="3pt"):
    env = "table*" if star else "table"
    return readable_table(
        "\n".join(
            [
                rf"\begin{{{env}}}[t]",
                r"\centering",
                r"\small",
                rf"\setlength{{\tabcolsep}}{{{sep}}}",
                rf"\begin{{tabular}}{{{cols}}}",
                *lines,
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                rf"\end{{{env}}}",
                "",
            ]
        )
    )


BLOCK_NOTE = r"\textbf{Bold}: best; \underline{underline}: second best, within each block. $^*$First 1,000 test questions of each dataset."
AGGREGATE_NOTE = "OOD: equal-weight mean over the five OOD datasets; Macro: equal-weight mean over all six datasets. Averages are computed before rounding."


CAL_NOTE = (
    r"Platt scaling (PS), temperature scaling (TS), and isotonic regression (ISO) see the score only; BWH (a bin-wise hybrid "
    r"of two logistic models) and ours also see the chain features and the judge outputs."
)


def calibrators_for(bb):
    caption = (
        f"Post-hoc calibrators on our probe's score, {dict(BACKBONES)[bb]}. " + CAL_NOTE
    )
    return calibrator_table(bb, caption, f"tab:calibration_{[b for b, _ in BACKBONES].index(bb) + 1}")


def significance_table():
    """Pairs per backbone where a paired bootstrap over test questions (1,000 resamples, 10 seeds) puts the 95% interval
    of the AUROC difference above zero: JIP vs. the best judge-free baseline, JIP + Judge vs. the best baseline
    (``analysis.significance``), and JIP + Judge vs. the stronger probe + judge fusion (``analysis.responses``)."""
    from chainuq.analysis.cache import ANALYSIS

    boot = json.loads((ANALYSIS / "responses.json").read_text())["bootstrap"]
    rows = []
    for name, key in ((r"\m vs.\ best judge-free", "A"), (r"\mJ vs.\ best baseline", "B"), (r"\mJ vs.\ fusion", "F")):
        win, lose = {}, 0
        for bb, _ in BACKBONES:
            win[bb] = 0
            for ds, _ in DATASETS:
                if key == "F":
                    lo, hi = boot[f"{bb}/{ds}"]["lo"], boot[f"{bb}/{ds}"]["hi"]
                else:
                    lo, hi = json.loads((results_dir() / "paper" / bb / f"{ds}_signif.json").read_text())[key]["ci95"]
                win[bb] += lo > 0
                lose += hi < 0
        assert lose == 0, f"{name}: a pair is significantly lower"
        rows.append(f"{name} & " + " & ".join(f"{win[bb]}/6" for bb, _ in BACKBONES) + f" & {sum(win.values())}/18 \\\\")
    lines = [r"\toprule", r"\textbf{Comparison} & Llama & Qwen3 & Ministral & All \\", r"\midrule", *rows]
    return wrap_table(
        lines,
        "l cccc",
        "Pairs on which our method is significantly better than the strongest competitor of its setting: paired "
        "bootstrap over test questions of the AUROC difference (1,000 resamples, 10 seeds), 95\\% interval above zero. "
        "Fusion: the stronger probe--judge fusion of each pair. No pair is significantly worse.",
        "tab:significance",
        sep="4pt",
    )


def test2_table():
    cols, lines = "ll ccc ccc", [r"\toprule"]
    lines.append(r" & & \multicolumn{3}{c}{\textbf{Judge-free}} & \multicolumn{3}{c}{\textbf{With external verifier}} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(r"\textbf{Backbone} & \textbf{Dataset} & \m & Best baseline & & \mJ & Best baseline & \\")
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
            lines.append(
                f"{bname.split('-')[0] if i == 0 else ''} & {dname.replace(' (ID)', '')} & {b(oa, va)} & {va:.3f} & {{\\small {na}}} & "
                f"{b(o, vb)} & {vb:.3f} & {{\\small {nb}}} \\\\"
            )
        lines.append(r"\midrule")
    lines = lines[:-1]
    return wrap_table(
        lines,
        cols,
        "AUROC on the fresh held-out Test-2 questions (single evaluation of the frozen pipeline). \\textbf{Bold}: \\m or \\mJ is at least as good as the strongest baseline of its setting.",
        "tab:test2",
        star=True,
        sep="4pt",
    )


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
    sampling = ms("gen[samples]")
    rows = [
        ("MCP", "--", Sc["mcp"]["ms_per_sample"], None),
        ("Perplexity", "--", Sc["ppl"]["ms_per_sample"], None),
        ("MTE", "--", Sc["mte"]["ms_per_sample"], None),
        ("CCP", "NLI", Sc["ccp"]["ms_per_sample"], Sc["ccp"]["peak_gpu_gb"]),
        ("CoT-UQ", "1 backbone call", 1000 * COTUQ_SECONDS / COTUQ_RESPONSES, None),
        ("Semantic Entropy", "10 samples + NLI", sampling + Sc["se_selfcheck_nli"]["ms_per_sample"], None),
        ("SelfCheckGPT", "10 samples + NLI", sampling + Sc["se_selfcheck_nli"]["ms_per_sample"], None),
        ("SAPLMA", "--", H["saplma"]["ms_per_sample"], H["saplma"]["peak_gpu_gb"]),
        ("Factoscope (adapted)", "--", H["factoscope"]["ms_per_sample"], H["factoscope"]["peak_gpu_gb"]),
        ("Lookback Lens", "--", H["lookback"]["ms_per_sample"], None),
        ("UHead", "--", H["uhead"]["ms_per_sample"], H["uhead"]["peak_gpu_gb"]),
        ("MeanPool-MLP", "--", H["meanpool-mlp"]["ms_per_sample"], H["meanpool-mlp"]["peak_gpu_gb"]),
        ("AttnPool-MLP", "--", H["attnpool-mlp"]["ms_per_sample"], H["attnpool-mlp"]["peak_gpu_gb"]),
        (r"\textbf{\m}", "--", H[FINAL_HEAD].get("ms_per_sample_full", H[FINAL_HEAD]["ms_per_sample"]) + rlog_ms, H[FINAL_HEAD]["peak_gpu_gb"]),
        ("Judge-Step", "1 judge call", judge_steps, None),
        ("Judge-Final", "1 judge call", judge_final, None),
        ("Reward Model", "1 RM forward", Sc["rm"]["ms_per_sample"], Sc["rm"]["peak_gpu_gb"]),
        (
            r"\textbf{\mJ}",
            "1 judge call",
            H[FINAL_HEAD].get("ms_per_sample_full", H[FINAL_HEAD]["ms_per_sample"]) + rlog_ms + judge_final,
            H[FINAL_HEAD]["peak_gpu_gb"],
        ),
    ]
    lines = [
        r"\toprule",
        r"\textbf{Method} & \textbf{Extra model calls} & \textbf{Latency (ms)} $\downarrow$ & \textbf{Probe GPU mem (GB)} $\downarrow$ \\",
        r"\midrule",
    ]
    for name, calls, lat, mem in rows:
        shade = r"\rowcolor{gray!20} " if name.startswith(r"\textbf{\m") else ""
        lines.append(f"{shade}{name} & {calls} & {fmt_ms(lat)} & {'--' if mem is None else f'{mem:.2f}'} \\\\")
    common = (ms("gen"), ms("features"))
    t1 = wrap_table(
        lines,
        "l l r r",
        (
            f"Post-generation cost per response on {bench['gpu'].replace('NVIDIA ', '')} (Llama, HotpotQA test), with batched "
            f"serving and probes run in batches of 64 on cached features. Shared and not included: greedy generation "
            f"({common[0]:.0f} ms) and, for white-box features, one forward pass ({common[1]:.0f} ms)."
        ),
        "tab:efficiency",
        sep="3pt",
    )
    lines = [r"\toprule", r"\textbf{Calibrator} & \textbf{Fit on validation (s)} $\downarrow$ & \textbf{Predict (ms/sample)} $\downarrow$ \\", r"\midrule"]
    for key, name in CALIBRATORS:
        shade = r"\rowcolor{gray!20} " if key == "rlog" else ""
        lines.append(f"{shade}{name} & {Cb[key]['fit_s']:.2f} & {fmt_ms(Cb[key]['ms_per_sample'])} \\\\")
    t2 = wrap_table(
        lines,
        "l r r",
        (
            f"Cost of the post-hoc calibrators on our probe's score (Llama, HotpotQA; {Cb['n_val']} validation and {Cb['n_test']} test responses, one CPU core). "
            "BWH (a bin-wise hybrid of two logistic models) and ours include building the chain features $u(r)$."
        ),
        "tab:efficiency_post",
        sep="4pt",
    )
    return t1, t2


def main():
    tdir = PAPER_DIR / "table"
    disc, unc = ["acc", "auroc", "prauc"], ["ece", "brier", "risk90"]
    names = dict(BACKBONES)
    (tdir / "result.tex").write_text(
        backbone_table(
            ["auroc", "prauc", "aurc"],
            r"\textbf{\m is the best method without a judge call on every backbone, and \mJ the best with one.} "
            r"Macro AUROC, AP, and AURC over the six datasets (10 seeds). \textbf{Bold}: best; \underline{underline}: second best, "
            r"within each block. $^*$First 1,000 test questions of each dataset. Per-dataset results: Appendix~\ref{sec:app_more_results}.",
            "tab:result",
        )
    )
    app = [
        method_table(
            "llama",
            disc,
            f"Discrimination results of {names['llama']} per dataset (10 seeds). " + BLOCK_NOTE,
            "tab:result_llama_detail",
        )
    ]
    app += [
        method_table(bb, disc, f"Discrimination results of {name} per dataset (10 seeds). " + BLOCK_NOTE, f"tab:result_{i}")
        for i, (bb, name) in enumerate(BACKBONES[1:], start=2)
    ]
    app += [
        method_table(
            bb,
            unc,
            f"Calibration and selective risk for {name}. Single-score baselines are calibrated with isotonic regression on validation; fusion baselines use their logistic regression. " + BLOCK_NOTE,
            f"tab:uq_{i}",
        )
        for i, (bb, name) in enumerate(BACKBONES, start=1)
    ]
    (tdir / "app_results.tex").write_text("\n".join(app))
    detail = calibrator_table("llama", f"Per-dataset post-hoc calibrators on our probe's score, {names['llama']}. " + CAL_NOTE, "tab:calibration_llama_detail")
    (tdir / "app_calibration.tex").write_text(detail + "\n" + calibrators_for("qwen8b") + "\n" + calibrators_for("ministral"))
    (tdir / "significance.tex").write_text(significance_table())
    (tdir / "test2.tex").write_text(test2_table())
    t1, t2 = efficiency_tables()
    (tdir / "efficiency.tex").write_text(t1)
    (tdir / "efficiency_post.tex").write_text(t2)
    print(tdir)


if __name__ == "__main__":
    main()
