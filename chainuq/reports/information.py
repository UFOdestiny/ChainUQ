"""Information-decomposition tables, in-text numbers, Figure 1, and the accepted-responses figure of the paper.

Inputs (all fit on validation and scored on test unless stated), under ``CHAINUQ_ANALYSIS``:
  runs/<tag>.jsonl                            Stage-2 variant runs (``analysis.variants``)
  information/decomposition_<head>[_*].json   information decomposition (``analysis.information``)
  information/routing_<head>.json             information-gain routing (``analysis.routing``)
  information/{babi_subtasks,unseen_domain}.json, responses.json    (``analysis.subtasks``, ``.unseen``, ``.responses``)
and ``results/<backbone>/eval*.json`` (``pipeline.evaluate``, ``analysis.fusion``).

Writes ``table/{decomposition,accounting_all,pools,pools_bb,attribution,bigjudge,ablation,grid}.tex``,
``table/information_numbers.tex`` and ``fig/{teaser,accepted}.pdf`` under ``CHAINUQ_PAPER_DIR``.

    python -m chainuq.reports.information
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.variants import EVIDENCE_A as EVA
from chainuq.analysis.variants import EVIDENCE_B as EVB
from chainuq.artifacts import results_dir
from chainuq.config import PAPER_DIR
from chainuq.registry import FINAL_HEAD, SOURCE_HEAD
from chainuq.reports.style import INK, ORANGE, TEAL, paper_plot_style

BBS = ["llama", "qwen8b", "ministral"]
DSS = ["hotpotqa", "musique", "strategyqa", "babi", "gsm8k", "math500"]
DSNAME = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "strategyqa": "StrategyQA", "babi": "bAbI", "gsm8k": "GSM8K", "math500": "MATH-500"}
GRAY = "#6C7880"
T = ANALYSIS / "information"
CHAINUQ_B, CHAINUQ_A = EVB + "@rlogs", EVA + "@rlogs"  # without the head key; the calibrator is R-Log with evidence-set selection
RESPONSES = ANALYSIS / "responses.json"
S3 = (2026, 2027, 2028)
macros = {}


def mac(name, value, fmt="{:.3f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value
    assert "nan" not in macros[name], f"macro {name} is nan"


def f3(v):
    return f"{v:.3f}"


# ---------------------------------------------------------------- official comparison
def official(split="test"):
    out = {}
    for bb in BBS:
        R = json.loads((results_dir() / bb / ("eval.json" if split == "test" else f"eval_{split}.json")).read_text())
        E = {}
        p = results_dir() / bb / ("eval_extra.json" if split == "test" else f"eval_extra_{split}.json")
        if p.exists():
            E = json.loads(p.read_text())["datasets"]
        cells = {}
        for ds, r in R["datasets"].items():
            row = {k: v["auroc"]["mean"] for k, v in {**r["tier_a"], **r["tier_b"]}.items()}
            row.update({k: v["auroc"]["mean"] for k, v in E.get(ds, {}).get("tier_b", {}).items()})
            cells[ds] = row
        out[bb] = cells
    return out


def main_macros():
    O = official()
    fus = ("fusion-saplma", "fusion-uhead")
    names = {"llama": "Llama", "qwen8b": "Qwen", "ministral": "Min"}
    B, A, J, F = [], [], [], []
    best_a = 0
    judge_free = ["mcp", "ppl", "mte", "ccp", "cotuq", "se", "selfcheck", "saplma", "factoscope", "lookback", "uhead", "meanpool-mlp", "attnpool-mlp"]
    for bb in BBS:
        cells = O[bb]
        b = np.mean([cells[d]["chainuq"] for d in DSS])
        f = max(np.mean([cells[d][k] for d in DSS]) for k in fus)
        mac(f"vB{names[bb]}", b)
        B.append(b), F.append(f), A.append(np.mean([cells[d]["chainuq-a"] for d in DSS]))
        J.append(np.mean([cells[d]["judge-final"] for d in DSS]))
        for d in DSS:
            c = cells[d]
            best_a += c["chainuq-a"] >= max(c[k] for k in judge_free if k in c)
    mac("vB", np.mean(B)), mac("vA", np.mean(A)), mac("vJ", np.mean(J)), mac("vFus", np.mean(F))
    mac("vAbest", str(best_a))
    # the strongest method without a judge or verifier call, per backbone (macro AUROC), averaged over backbones
    mac("vFreeBest", np.mean([max(np.mean([O[bb][d][k] for d in DSS]) for k in judge_free if k in O[bb]["hotpotqa"]) for bb in BBS]))
    O2 = official("test2")
    mac("vTwoB", np.mean([np.mean([O2[bb][d]["chainuq"] for d in O2[bb]]) for bb in BBS]))
    # the stronger of the two probe--judge fusion baselines of each pair, as for the test split
    mac("vTwoF", np.mean([max(O2[bb][d][k] for k in fus) for bb in BBS for d in O2[bb]]))
    return O


# ---------------------------------------------------------------- decomposition
def info_rows(path):
    rows = json.loads(path.read_text())
    cell = defaultdict(list)
    for r in rows:
        cell[r["bb"], r["ds"]].append(r)
    out = {}
    for k, rr in cell.items():
        m = {key: float(np.mean([r[key] for r in rr])) for key in rr[0] if isinstance(rr[0][key], float)}
        m["I_AgJ"] = m["ce_J"] - m["ce_AJ"]
        m["I_JgA"] = m["ce_A"] - m["ce_AJ"]
        m["dR"] = m["ce_J"] - m["ce_A"]
        m["acc_part"] = m["frac_acc"] * m["I_AgJ_acc"]
        m["rej_part"] = m["frac_rej"] * m["I_AgJ_rej"]
        out[k] = m
    return out


def accounting_table(O):
    I = info_rows(T / f"decomposition_{FINAL_HEAD}.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"\textbf{Dataset} & $\hat I(y;A\mid J)$ & \emph{accepted} & $\hat I(y;J\mid A)$ & $\Delta$AUROC \\",
        r"\midrule",
    ]
    agree, pos, acc_parts, tot_parts = 0, 0, [], []
    for d in DSS:
        ms = [I[bb, d] for bb in BBS]
        dauc = np.mean([O[bb][d]["chainuq-a"] - O[bb][d]["judge-final"] for bb in BBS])
        np.sum([m["acc_part"] for m in ms]) / np.sum([m["acc_part"] + m["rej_part"] for m in ms])
        lines.append(
            f"{DSNAME[d]} & {np.mean([m['I_AgJ'] for m in ms]):.3f} & {np.mean([m['acc_part'] for m in ms]):.3f} & "
            f"{np.mean([m['I_JgA'] for m in ms]):.3f} & {dauc:+.3f} \\\\"
        )
        for bb in BBS:
            m = I[bb, d]
            agree += np.sign(m["dR"]) == np.sign(O[bb][d]["chainuq-a"] - O[bb][d]["judge-final"])
            pos += m["I_AgJ"] > 0
            acc_parts.append(m["acc_part"]), tot_parts.append(m["acc_part"] + m["rej_part"])
    allm = list(I.values())
    lines += [
        r"\midrule",
        f"Mean & {np.mean([m['I_AgJ'] for m in allm]):.3f} & {np.mean([m['acc_part'] for m in allm]):.3f} & "
        f"{np.mean([m['I_JgA'] for m in allm]):.3f} & "
        f"{np.mean([O[bb][d]['chainuq-a'] - O[bb][d]['judge-final'] for bb in BBS for d in DSS]):+.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Information decomposition (nats; Eq.~\eqref{eq:accounting}), mean over the three backbones "
        r"and 10 seeds. \emph{Accepted}: the part of $\hat I(y;A\mid J)$ contributed by the responses the judge accepts "
        r"($v\geq0.5$), $\mathbb{P}(v{\geq}0.5)\,\hat I(y;A\mid J,v{\geq}0.5)$. "
        r"$\Delta$AUROC: \mA minus Judge-Final.}",
        r"\label{tab:decomposition}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "decomposition.tex").write_text("\n".join(lines) + "\n")
    mac("vSignAgree", str(int(agree))), mac("vIposPairs", str(int(pos)))
    mac("vAccShare", f"{100 * np.sum(acc_parts) / np.sum(tot_parts):.0f}", "{}")
    return I


# ---------------------------------------------------------------- R-Log runs
def runs(*tags, split="test"):
    out = defaultdict(dict)
    for tag in tags:
        p = ANALYSIS / "runs" / f"{tag}.jsonl"
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r["res"] and split in r["res"]:
                v, bb, ds, seed = r["job"]
                out[v][bb, ds, seed] = r["res"][split]
    return out


def macro(R, variant, seeds=None, metric="auroc", bbs=BBS):
    vals = []
    for bb in bbs:
        per = [np.mean([R[variant][bb, d, s][metric] for s in (seeds or range(2026, 2036)) if (bb, d, s) in R[variant]]) for d in DSS]
        vals.append(np.mean(per))
    return float(np.mean(vals))


def pools_table():
    R = runs("attribution", "target_pool_controls")
    heads = [
        ("No PCS", "chainuq-wo-pcs"),
        ("PCS, source only", SOURCE_HEAD),
        ("+ 6k, step labels only", "chainuq-pool-steps"),
        ("+ 6k, verdict only", "chainuq-pool-verdict"),
        (r"+ 6k unlabeled", FINAL_HEAD),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{l ccc cc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Decomposition (nats)}} & \multicolumn{2}{c}{\textbf{AUROC}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}",
        r"\textbf{Head} & $\widehat{\mathrm{CE}}(L{\mid}A)$ & $\hat I(y;J{\mid}A)$ & $\hat I(y;A{\mid}J)$ & \mA & \m \\",
        r"\midrule",
    ]
    vals = {}
    for name, h in heads:
        I = info_rows(T / f"decomposition_{h}_s3.json")
        m = list(I.values())
        a, b = macro(R, f"{CHAINUQ_A}#{h}", S3), macro(R, f"{CHAINUQ_B}#{h}", S3)
        vals[h] = (np.mean([x["ce_Jq"] for x in m]), np.mean([x["I_JgA"] for x in m]), np.mean([x["I_AgJ"] for x in m]), a, b)
        lines.append(f"{name} & " + " & ".join(f"{x:.3f}" for x in vals[h]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{\textbf{Better imitation of the judge lowers its residual information and raises judge-free AUROC.} "
        r"Held-out imitation loss, decomposition terms (nats), and AUROC (macro over datasets and backbones, 3 seeds). "
        r"Unlabeled: 6k questions per OOD domain, with both imitation targets unless noted.}",
        r"\label{tab:pools}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "pools.tex").write_text("\n".join(lines) + "\n")
    no, six = vals["chainuq-wo-pcs"], vals[FINAL_HEAD]
    mac("vPoolANo", no[3]), mac("vPoolASix", six[3])


def attribution_table():
    R = runs("attribution")
    CB, CA = "B-u_reason-u_token-u_judge-vhat+ptrue@rlogs", "A-u_reason-u_token-vhat@rlogs"
    PB, PA = "B-u_reason-u_token-u_judge+ptrue+vq@rlogs", "A-u_reason-u_token+vq@rlogs"
    rows = [
        (r"\multicolumn{4}{l}{\emph{With the judge}} \\", None),
        (
            "Probe + judge (LR)",
            {"saplma": "S+verdict+ptrue@lrsel#saplma", "uhead": "S+verdict+ptrue@lrsel#uhead", "ours": f"S+verdict+ptrue@lrsel#{SOURCE_HEAD}"},
        ),
        ("+ chain features (R-Log)", {"saplma": f"{CB}#saplma", "uhead": f"{CB}#uhead", "ours": f"{CB}#{SOURCE_HEAD}"}),
        (r"+ PCS $\hat v$, labeled source", {"uhead": f"{PB}#uhead-pcs", "ours": f"{PB}#{SOURCE_HEAD}"}),
        (r"+ unlabeled imitation", {"uhead": f"{PB}#uhead-pcs-pool6k", "ours": f"{PB}#{FINAL_HEAD}"}),
        (r"+ second probe (\m)", {"uhead": f"{CHAINUQ_B}#uhead-pcs-pool6k", "ours": f"{CHAINUQ_B}#{FINAL_HEAD}"}),
        (r"\midrule \multicolumn{4}{l}{\emph{Judge-free}} \\", None),
        ("Probe alone (ISO)", {"saplma": "S@iso#saplma", "uhead": "S@iso#uhead", "ours": f"S@iso#{SOURCE_HEAD}"}),
        ("+ chain features (R-Log)", {"saplma": f"{CA}#saplma", "uhead": f"{CA}#uhead", "ours": f"{CA}#{SOURCE_HEAD}"}),
        (r"+ PCS $\hat v$, labeled source", {"uhead": f"{PA}#uhead-pcs", "ours": f"{PA}#{SOURCE_HEAD}"}),
        (r"+ unlabeled imitation", {"uhead": f"{PA}#uhead-pcs-pool6k", "ours": f"{PA}#{FINAL_HEAD}"}),
        (r"+ second probe (\mA)", {"uhead": f"{CHAINUQ_A}#uhead-pcs-pool6k", "ours": f"{CHAINUQ_A}#{FINAL_HEAD}"}),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"\textbf{Evidence} & \textbf{SAPLMA} & \textbf{UHead} & \textbf{Ours} \\",
        r"\midrule",
    ]
    for name, spec in rows:
        if spec is None:
            lines.append(name)
            continue
        cells = []
        for col in ("saplma", "uhead", "ours"):
            v = spec.get(col)
            if v is None or v not in R:
                cells.append("--")
                continue
            seeds = S3 if "uhead-pcs" in v else None
            cells.append(f3(macro(R, v, seeds)))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Chain features and label-free imitation give the largest gains, for UHead as for our head.} "
        r"Macro AUROC over datasets and backbones (10 seeds; UHead with PCS and the source-only head: 3 seeds); each row "
        r"adds to the one above. LR: logistic regression; ISO: isotonic regression.}",
        r"\label{tab:attribution}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "attribution.tex").write_text("\n".join(lines) + "\n")


def judge_auroc(judge):
    """AUROC of the judge alone: logistic regression on its verdict and the log-odds of P(true), fit on validation;
    macro over datasets and backbones, test split."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from chainuq.analysis.cache import cache_path
    from chainuq.evaluation import verdict_score
    from chainuq.samples import load_judge_task

    def feats(bb, d, sp):
        cell = pickle.load(open(cache_path(bb, d, sp), "rb"))
        rec = load_judge_task(judge, "final_blind", bb, d, sp)
        v = np.nan_to_num(np.asarray([verdict_score(rec.get(u, {})) for u in cell["uids"]], float), nan=0.5)
        pt = np.asarray([rec.get(u, {}).get("p_true") or np.nan for u in cell["uids"]], float)
        pt = np.clip(np.nan_to_num(pt, nan=0.5), 1e-12, 1 - 1e-12)
        return np.column_stack([v, np.log(pt / (1 - pt))]), cell["y"]

    vals = []
    for bb in BBS:
        per = []
        for d in DSS:
            (Xv, yv), (Xt, yt) = feats(bb, d, "validation"), feats(bb, d, "test")
            per.append(roc_auc_score(yt, LogisticRegression(C=1.0, max_iter=1000).fit(Xv, yv).predict_proba(Xt)[:, 1]))
        vals.append(np.mean(per))
    return float(np.mean(vals))


def bigjudge_table():
    R20, R120 = runs("attribution"), runs("judge_strength")
    I20, I120 = info_rows(T / f"decomposition_{FINAL_HEAD}.json"), info_rows(T / f"decomposition_{FINAL_HEAD}_gptoss120b.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"\textbf{Judge} & Judge only & Fusion & \m & $\hat I(y;A\mid J)$ \\",
        r"\midrule",
    ]
    out = {}
    RQ, IQ = runs("cross_family_judge"), info_rows(T / f"decomposition_{FINAL_HEAD}_qwen3judge.json")
    keys = {"gpt-oss-20b": "gptoss", "gpt-oss-120b": "gptoss120b", "Qwen3-8B, thinking": "qwen3judge"}
    for name, R, I in (("gpt-oss-20b", R20, I20), ("gpt-oss-120b", R120, I120), ("Qwen3-8B, thinking", RQ, IQ)):
        jf = judge_auroc(keys[name])
        fus = max(macro(R, f"S+verdict+ptrue@lrsel#{h}") for h in ("saplma", "uhead"))
        b = macro(R, f"{CHAINUQ_B}#{FINAL_HEAD}")
        out[name] = (jf, fus, b)
        lines.append(f"{name} & {jf:.3f} & {fus:.3f} & {b:.3f} & {np.mean([I[k]['I_AgJ'] for k in I]):.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{The internal states add to every judge.} Judge swap at inference (macro AUROC, 10 seeds); the head "
        r"keeps its gpt-oss-20b training. Judge only: logistic regression on $v$ and $\mathrm{logit}\,p_{\mathrm{true}}$; "
        r"Fusion: the stronger of SAPLMA and UHead with the same judge outputs.}",
        r"\label{tab:bigjudge}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "bigjudge.tex").write_text("\n".join(lines) + "\n")
    mac("vBigJsmall", out["gpt-oss-20b"][0])


# ---------------------------------------------------------------- routing and Figure 1
def routing():
    rows = json.loads((T / f"routing_{FINAL_HEAD}.json").read_text())
    # the method routes by EIG with model-consistent judge-free scores (Proposition 3); "eig" keeps ChainUQ-A instead
    R = {k: np.mean([r["auroc"][src] for r in rows], 0) for k, src in (("eig", "eig"), ("conf", "conf"), ("random", "random"))}
    rec = lambda c, i: 100 * (c[i] - c[0]) / (c[-1] - c[0])
    mac("vRouteHalf", f"{rec(R['eig'], 5):.0f}", "{}"), mac("vRouteRand", f"{rec(R['random'], 5):.0f}", "{}")
    mac("vRouteHalfAUC", R["eig"][5])
    O_ = official()
    fus_ = np.mean([max(np.mean([O_[bb][d][k] for d in DSS]) for k in ("fusion-saplma", "fusion-uhead")) for bb in BBS])
    mac("vRouteCross", str(int(10 * next(i for i, v in enumerate(R["eig"]) if v > fus_))), "{}")
    gap = np.abs(np.mean([r["auroc"]["eig"] for r in rows], 0) - np.mean([r["auroc"]["eig-mix"] for r in rows], 0)).max()
    mac("vRouteMixGap", gap)
    return R


def reliability():
    """Accuracy per calibrated head-confidence bin, split by the judge verdict (Llama, seed 2026, pooled test sets)."""
    from chainuq import calibration as C
    from chainuq.analysis.variants import blocks

    conf, v, y = [], [], []
    for d in DSS:
        bv, bt = blocks("llama", d, "validation", FINAL_HEAD, 2026), blocks("llama", d, "test", FINAL_HEAD, 2026)
        m = C.Isotonic().fit(bv["s"], bv["y"])
        conf.append(m.predict(bt["s"])), v.append(np.nan_to_num(bt["verdict"][:, 0], nan=0.5)), y.append(bt["y"])
    conf, v, y = map(np.concatenate, (conf, v, y))
    edges = np.linspace(0, 1, 6)
    out = {}
    for name, sel in (("high", v >= 0.5), ("low", v < 0.5)):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (conf >= lo) & (conf < hi + (hi == 1))
            if m.sum() >= 20:
                xs.append((lo + hi) / 2), ys.append(float(y[m].mean()))
        out[name] = (xs, ys)
    return out


def teaser(I, route):
    paper_plot_style()
    rel = reliability()
    height, width, gap = 132, 160, 12
    fig = plt.figure(figsize=(7, height / 72))
    fw, fh = fig.get_size_inches() * 72
    titles = ("Confidence hides risk", "Who knows what", "Calling the judge selectively")
    axes = []
    for i in range(3):
        origin = i * (width + gap)
        left, aw = (31, 125) if i == 0 else (37, 119)
        ax = fig.add_axes([(origin + left) / fw, 29 / fh, aw / fw, 80 / fh])
        ax.tick_params(length=2.5, pad=2, direction="in")
        ax.xaxis.labelpad = ax.yaxis.labelpad = 3
        fig.text((origin + 1) / fw, 123 / fh, f"({chr(97 + i)}) {titles[i]}", weight="bold", color=INK, va="center")
        axes.append(ax)
    a = axes[0]
    for name, color, marker, ls in (("low", ORANGE, "s", "--"), ("high", TEAL, "o", "-")):
        a.plot(*rel[name], color=color, marker=marker, ms=3.8, lw=1.25, ls=ls, markeredgecolor="white", markeredgewidth=0.4)
    a.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0.5, 1], yticks=[0, 0.5, 1], xlabel="Head confidence", ylabel="Accuracy")
    a.text(0.04, 0.92, r"Judge $\geq 0.5$", transform=a.transAxes, color=TEAL, va="top")
    a.text(0.38, 0.04, r"Judge $<0.5$", transform=a.transAxes, color=ORANGE, va="bottom")
    b = axes[1]
    x = np.arange(len(DSS))
    pa = [np.mean([I[bb, d]["I_AgJ"] for bb in BBS]) for d in DSS]
    pj = [np.mean([I[bb, d]["I_JgA"] for bb in BBS]) for d in DSS]
    b.bar(x - 0.19, pa, width=0.36, color=TEAL)
    b.bar(x + 0.19, pj, width=0.36, color=ORANGE)
    b.set(xticks=x, xticklabels=["HQA", "MSQ", "SQA", "bAbI", "GSM", "MATH"], ylabel="Information (nats)")
    b.tick_params(axis="x", length=0, labelsize=6.5)
    b.legend(
        handles=[Patch(color=TEAL, label=r"$I(y;A\mid J)$: probe beyond judge"), Patch(color=ORANGE, label=r"$I(y;J\mid A)$: judge beyond probe")],
        loc="upper left",
        frameon=False,
        handlelength=0.8,
        handletextpad=0.3,
        borderaxespad=0.1,
        labelspacing=0.15,
        fontsize=7.5,
    )
    b.set_ylim(0, max(max(pa), max(pj)) * 1.45)
    c = axes[2]
    xs = 100 * np.linspace(0, 1, len(route["eig"]))
    c.plot(xs, route["random"], color=GRAY, lw=1.1, ls="--", marker="o", ms=2.6, markerfacecolor="white")
    c.plot(xs, route["eig"], color=TEAL, lw=1.3, marker="D", ms=2.8, markeredgecolor="white", markeredgewidth=0.3)
    # baselines at their judge budget: the probes call no judge, the judge and the fusions call it on every response
    O = official()
    mac_ = lambda k: np.mean([np.mean([O[bb][d][k] for d in DSS]) for bb in BBS])
    probes = ("saplma", "uhead", "factoscope", "lookback", "meanpool-mlp", "attnpool-mlp")
    pts = [
        (0, max(mac_(k) for k in probes if k in O["llama"]["hotpotqa"]), "Best probe", "v"),
        (100, mac_("judge-final"), "Judge-Final", "s"),
        (100, max(mac_("fusion-saplma"), mac_("fusion-uhead")), "Probe + judge fusion", "^"),
    ]
    for x0, y0, name, m in pts:
        c.plot([x0], [y0], linestyle="none", marker=m, ms=4.2, color=ORANGE, markeredgecolor="white", markeredgewidth=0.4, clip_on=False, zorder=4)
        if name == "Probe + judge fusion":  # labeled at the left end of its reference line, clear of the curves
            c.text(97, y0 - 0.0055, "Probe + judge", fontsize=6.4, color=ORANGE, va="center", ha="right")
            continue
        dy = 0
        c.text(x0 + (5 if x0 == 0 else -5), y0 + dy, name, fontsize=6.4, color=ORANGE, va="center", ha="left" if x0 == 0 else "right")
    c.axhline(max(mac_("fusion-saplma"), mac_("fusion-uhead")), color=ORANGE, lw=0.6, ls=":", zorder=1)
    lo, hi = min(y for _, y, _, _ in pts), route["eig"][-1]
    c.set(xlim=(-4, 104), xticks=[0, 50, 100], ylim=(lo - 0.006, hi + 0.006), xlabel="Responses sent to the judge (%)", ylabel="AUROC")
    key = lambda label, color, marker, hollow=False: Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markersize=4.5,
        markeredgecolor=color,
        markeredgewidth=0.7,
        markerfacecolor="white" if hollow else color,
        label=label,
    )
    c.legend(
        handles=[key("ChainUQ, info.-gain routing", TEAL, "D"), key("ChainUQ, random routing", GRAY, "o", hollow=True)],
        loc="lower center",
        bbox_to_anchor=(0.56, 0.12),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        handlelength=0.7,
        handletextpad=0.25,
        borderpad=0.15,
        borderaxespad=0.2,
        labelspacing=0.12,
        fontsize=7,
    )
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    out = PAPER_DIR / "fig" / "teaser.pdf"
    fig.savefig(out)
    print(out)


def accounting_all_table(O):
    I = info_rows(T / f"decomposition_{FINAL_HEAD}.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{ll ccc c}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Dataset} & $\hat I(y;A{\mid}J)$ & \emph{accepted} & $\hat I(y;J{\mid}A)$ & $\Delta$AUROC \\",
        r"\midrule",
    ]
    names = {"llama": "Llama", "qwen8b": "Qwen3", "ministral": "Ministral"}
    for bi, bb in enumerate(BBS):
        for di, d in enumerate(DSS):
            m = I[bb, d]
            tot = m["acc_part"] + m["rej_part"]
            f"{100 * m['acc_part'] / tot:.0f}\\%" if tot > 0 else "--"
            lines.append(
                f"{names[bb] if di == 0 else ''} & {DSNAME[d]} & {m['I_AgJ']:.3f} & {m['acc_part']:.3f} & {m['I_JgA']:.3f} & "
                f"{O[bb][d]['chainuq-a'] - O[bb][d]['judge-final']:+.3f} \\\\"
            )
        if bi < 2:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Information decomposition per backbone and dataset (nats; 10 seeds). Columns as in "
        r"Table~\ref{tab:decomposition}.}",
        r"\label{tab:accounting_all}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "accounting_all.tex").write_text("\n".join(lines) + "\n")


def pools_bb_table():
    R = runs("attribution")
    heads = [("No PCS", "chainuq-wo-pcs"), ("PCS, source", SOURCE_HEAD), ("+ 2k unlab.", "chainuq-pool2k"), (r"+ 6k unlab.", FINAL_HEAD)]
    names = {"llama": "Llama", "qwen8b": "Qwen3", "ministral": "Ministral"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{ll ccc cc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Head} & $\widehat{\mathrm{CE}}(L{\mid}A)$ & $\hat I(y;J{\mid}A)$ & $\hat I(y;A{\mid}J)$ & \mA & \m \\",
        r"\midrule",
    ]
    for bi, bb in enumerate(BBS):
        for hi, (name, h) in enumerate(heads):
            I = [m for k, m in info_rows(T / f"decomposition_{h}_s3.json").items() if k[0] == bb]
            a = macro(R, f"{CHAINUQ_A}#{h}", S3, bbs=[bb])
            b = macro(R, f"{CHAINUQ_B}#{h}", S3, bbs=[bb])
            lines.append(
                f"{names[bb] if hi == 0 else ''} & {name} & {np.mean([m['ce_Jq'] for m in I]):.3f} & "
                f"{np.mean([m['I_JgA'] for m in I]):.3f} & {np.mean([m['I_AgJ'] for m in I]):.3f} & {a:.3f} & {b:.3f} \\\\"
            )
        if bi < 2:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Label-free imitation per backbone (macro over datasets, seeds 2026--2028). Columns as in "
        r"Table~\ref{tab:pools}.}",
        r"\label{tab:pools_bb}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "pools_bb.tex").write_text("\n".join(lines) + "\n")


def ablation_table():
    R = runs("attribution")
    H = FINAL_HEAD
    rows = [
        (r"\m / \mA", f"{EVB}@rlogs#{H}", f"{EVA}@rlogs#{H}"),
        ("R-Log without evidence selection", f"{EVB}@rlog#{H}", f"{EVA}@rlog#{H}"),
        ("LR, same inputs", f"{EVB}@lrsel#{H}", f"{EVA}@lrsel#{H}"),
        ("GBM, same inputs", f"{EVB}@gbm#{H}", f"{EVA}@gbm#{H}"),
        (r"$-$ $p_{\mathrm{true}}$", f"{EVB}-ptrue@rlogs#{H}", None),
        (r"$-$ imitation outputs $\hat v,\,q_\theta(L)$", f"{EVB}-vhat-vq@rlogs#{H}", f"{EVA}-vhat-vq@rlogs#{H}"),
        (r"$-$ second probe", f"B-u_reason-u_token-u_judge+ptrue+vq@rlogs#{H}", f"A-u_reason-u_token+vq@rlogs#{H}"),
        (r"$-$ $u_{\text{span}}$", f"{EVB}-span@rlogs#{H}", f"{EVA}-span@rlogs#{H}"),
        (r"$-$ $u_{\text{gap}}$", f"{EVB}-u_gap@rlogs#{H}", f"{EVA}-u_gap@rlogs#{H}"),
        ("Full evidence", f"B+ptrue+head:saplma+vq@rlogs#{H}", f"A+head:saplma+vq@rlogs#{H}"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{With judge}} & \multicolumn{3}{c}{\textbf{Judge-free}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"\textbf{Variant} & AUROC$\uparrow$ & NLL$\downarrow$ & ECE$\downarrow$ & AUROC$\uparrow$ & NLL$\downarrow$ & ECE$\downarrow$ \\",
        r"\midrule",
    ]
    for name, vb, va in rows:
        cells = []
        for v in (vb, va):
            for metric in ("auroc", "nll", "ece"):
                cells.append(f3(macro(R, v, metric=metric)) if v and v in R else "--")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Evidence and calibrator ablations of the final method (macro over datasets and backbones, "
        r"10 seeds). Full evidence adds the step-score summaries, whole-response token statistics, and (with the "
        r"judge) the judge's step labels.}",
        r"\label{tab:ablation}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "ablation.tex").write_text("\n".join(lines) + "\n")


def response_macros():
    R = json.loads(RESPONSES.read_text())
    mac("vSigWin", str(sum(v["sig"] == "+" for v in R["bootstrap"].values())))
    mac("vSilent", f"{100 * np.mean([v['silent_frac'] for v in R['silent'].values()]):.0f}", "{}")
    O2 = official("test2")
    mac("vTwoJ", np.mean([np.mean([O2[bb][d]["judge-final"] for d in O2[bb]]) for bb in BBS]))
    mac("vTwoA", np.mean([np.mean([O2[bb][d]["chainuq-a"] for d in O2[bb]]) for bb in BBS]))


def decomposition_macros():
    """Bootstrap intervals of the two decomposition terms, and the accepted share of I(y; A | J) under GBM."""
    rows = json.loads((T / f"decomposition_{FINAL_HEAD}.json").read_text())
    cell = defaultdict(list)
    for r in rows:
        cell[r["bb"], r["ds"]].append(r)
    for key, name in (("boot_I_AgJ", "IAgJ"), ("boot_I_JgA", "IJgA")):
        per = np.array([np.mean([r[key] for r in rr], 0) for rr in cell.values()])  # [cells, boot]
        mac(f"v{name}", per.mean(0).mean()), mac(f"v{name}Lo", np.percentile(per.mean(0), 2.5))
        mac(f"v{name}Hi", np.percentile(per.mean(0), 97.5))
    G = info_rows(T / f"decomposition_{FINAL_HEAD}_gbm.json")
    acc = sum(m["acc_part"] for m in G.values()) / sum(m["acc_part"] + m["rej_part"] for m in G.values())
    mac("vAccShareGBM", f"{100 * acc:.0f}", "{}")


def bound_macros():
    """Proposition 1: the quadratic bound and its calibrated forms."""
    rows = json.loads((T / f"decomposition_{FINAL_HEAD}.json").read_text())
    mac("vQuadBound", np.mean([r["quad_bound"] for r in rows])), mac("vQuadI", np.mean([r["I_JgA_onehot"] for r in rows]))
    mac("vGiniBound", np.mean([r["gini_bound"] for r in rows])), mac("vQuadCal", np.mean([r["quad_cal"] for r in rows]))
    mac("vHalfB", np.mean([r["half_range_B"] for r in rows]), "{:.2f}")
    mac("vQCalib", np.mean([r["q_calib_ece"] for r in rows])), mac("vQLevelEce", np.mean([r["q_level_ece"] for r in rows]))
    full = np.mean([r["ce_A"] - r["ce_AJ"] for r in rows])  # I(J -> y | A) for the full judge output
    mac("vWithinLevel", full - np.mean([r["I_JgA_onehot"] for r in rows]))
    mac("vLevelShare", f"{100 * np.mean([r['I_JgA_onehot'] for r in rows]) / full:.0f}", "{}")


def judge_effort_macros():
    """gpt-oss-120b at high reasoning effort on GSM8K and MATH-500."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from chainuq.analysis.cache import cache_path
    from chainuq.evaluation import verdict_score
    from chainuq.samples import load_judge_task

    math = ["gsm8k", "math500"]
    vals = []
    for bb in BBS:
        for d in math:
            xs = {}
            for sp in ("validation", "test"):
                cell = pickle.load(open(cache_path(bb, d, sp), "rb"))
                rec = load_judge_task("gptoss120b-high", "final_blind", bb, d, sp)
                v = np.nan_to_num(np.asarray([verdict_score(rec.get(u, {})) for u in cell["uids"]], float), nan=0.5)
                pt = np.clip(np.nan_to_num(np.asarray([rec.get(u, {}).get("p_true") or np.nan for u in cell["uids"]], float), nan=0.5), 1e-12, 1 - 1e-12)
                xs[sp] = (np.column_stack([v, np.log(pt / (1 - pt))]), cell["y"])
            m = LogisticRegression(max_iter=1000).fit(*xs["validation"])
            vals.append(roc_auc_score(xs["test"][1], m.predict_proba(xs["test"][0])[:, 1]))
    mac("vHighJ", float(np.mean(vals)))
    R = runs("judge_reasoning_effort")
    v = f"{CHAINUQ_B}#{FINAL_HEAD}"
    mac("vHighB", float(np.mean([np.mean([R[v][bb, d, s]["auroc"] for s in range(2026, 2036)]) for bb in BBS for d in math])))


def subtask_macros():
    B = json.loads((T / "babi_subtasks.json").read_text())
    mac("vBabiWithin", np.mean([B[bb]["ChainUQ"]["within"] for bb in BBS]))
    mac("vBabiWithinFus", np.mean([B[bb]["SAPLMA + J"]["within"] for bb in BBS]))
    mac("vBabiWithinJ", np.mean([B[bb]["Judge-Final"]["within"] for bb in BBS]))
    mac("vBabiPrior", np.mean([B[bb]["Task prior"]["pooled"] for bb in BBS]))


def distill_macros():
    """SAPLMA distilled from the judge on the unlabeled pools, and SAPLMA itself, each fused with the judge."""
    mac("vDistFus", macro(runs("probe_distillation"), "S+verdict+ptrue@lrsel#saplma-pool6k-pseudo", S3))
    mac("vSapFus", macro(runs("attribution"), "S+verdict+ptrue@lrsel#saplma", S3))


def grid_table():
    """Label grid: the head and the second probe (SAPLMA) see the same fraction of the source labels."""
    G, R = runs("calibration_grid"), runs("calibration_grid_extended")
    bbs = ["llama", "qwen8b"]
    heads = [("No PCS", "chainuq-wo-pcs"), ("PCS, source", SOURCE_HEAD), ("+ 6k unlabeled", FINAL_HEAD)]

    def cell(frac, h, judge=False):
        if frac == "100":
            return macro(G, f"{CHAINUQ_B if judge else CHAINUQ_A}#{h}", S3, bbs=bbs)
        ev = (EVB if judge else EVA).replace("head:saplma", f"head:saplma-lf{frac}")
        return macro(R, f"{ev}@rlogs#{h}-lf{frac}", S3, bbs=bbs)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"\textbf{Source labels} & " + " & ".join(n for n, _ in heads) + r" \\",
        r"\midrule",
    ]
    val = {}
    for frac in ("01", "05", "25", "100"):
        val[frac] = [cell(frac, h) for _, h in heads]
        lines.append(f"{int(frac)}\\% & " + " & ".join(f3(v) for v in val[frac]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Judge-free AUROC of \mA when the head and the second probe see 1--100\% of the HotpotQA labels; "
        r"R-Log uses the target validation labels in every row (Llama and Qwen3, 3 seeds).}",
        r"\label{tab:grid}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "grid.tex").write_text("\n".join(lines) + "\n")


def comparison_macros():
    O = official()
    fus = ("fusion-saplma", "fusion-uhead")
    mac("vMarginPos", str(sum(O[bb][d]["chainuq"] - max(O[bb][d][f] for f in fus) > 0 for bb in BBS for d in DSS)))
    mac("vZeroJ", np.mean([np.mean([O[bb][d]["judge-final"] for d in DSS[1:]]) for bb in BBS]))


def unseen_macros():
    """Unseen domain (no target information) and leave-one-domain-out imitation."""
    U = json.loads((T / "unseen_domain.json").read_text())["unseen"]
    mac("vUnseenB", np.mean([r["auroc"] for r in U if r["variant"].startswith("B")]))
    R = runs("attribution", "leave_one_domain_out")
    wins = 0
    for bb in BBS:
        for d in DSS[1:]:
            lodo, src = (np.mean([R[f"{CHAINUQ_A}#{h}"][bb, d, s]["auroc"] for s in S3]) for h in (f"{FINAL_HEAD}-lodo-{d}", SOURCE_HEAD))
            wins += lodo > src
    mac("vLodoWins", str(wins))


def think_macros():
    """Thinking-mode Qwen3-8B (internal states include the thinking trace; the judge reads the visible response)."""
    R, bb = runs("thinking"), ["qwen8b-think"]
    S5 = range(2026, 2031)
    head = "chainuq-pool2k"  # 6k pools were not built for the thinking backbone
    mac("vThinkB", macro(R, f"{CHAINUQ_B}#{head}", S5, bbs=bb))
    mac("vThinkFus", max(macro(R, f"S+verdict+ptrue@lrsel#{h}", S5, bbs=bb) for h in ("saplma", "uhead")))
    I = json.loads((T / "decomposition_thinking.json").read_text())
    mac("vThinkJ", np.mean([np.mean([r["auc_J"] for r in I if r["ds"] == d]) for d in DSS]))
    # the same judge shown the thinking trace as well (judge key gptoss-trace; the head still imitates the plain judge)
    R2 = runs("judge_trace")
    mac("vTraceB", macro(R2, f"{CHAINUQ_B}#{head}", S5, bbs=bb))
    mac("vTraceFus", max(macro(R2, f"S+verdict+ptrue@lrsel#{h}", S5, bbs=bb) for h in ("saplma", "uhead")))


def accepted_figure():
    """Errors the judge lets through: AUROC on the responses the judge accepts (v >= 0.5), per dataset, mean over
    the three backbones (judge score vs. the head's raw score vs. ChainUQ)."""
    paper_plot_style()
    S = json.loads(RESPONSES.read_text())["silent"]
    series = [("Judge-Final", "judge", ORANGE), ("Head (raw score)", "head", "#9AA4AB"), (r"\textsc{ChainUQ}", "chainuq", TEAL)]
    vals = {k: [np.mean([S[f"{bb}/{d}"]["accepted"][k] for bb in BBS]) for d in DSS] for _, k, _ in series}
    fig = plt.figure(figsize=(3.3, 1.55))
    ax = fig.add_axes([0.12, 0.2, 0.86, 0.66])
    x, w = np.arange(len(DSS)), 0.26
    for i, (name, k, c) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals[k], width=w - 0.03, color=c, label=name.replace(r"\textsc{ChainUQ}", "ChainUQ"))
    ax.set_xticks(x), ax.set_xticklabels([DSNAME[d] for d in DSS], fontsize=6.6)
    ax.set_ylim(0.5, 1.0), ax.set_ylabel("AUROC (accepted)", fontsize=7)
    ax.tick_params(length=2, pad=2, direction="in", labelsize=6.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.4, handlelength=0.9, columnspacing=0.9, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(PAPER_DIR / "fig" / "accepted.pdf")
    mac("vAccJudge", np.mean(vals["judge"])), mac("vAccOurs", np.mean(vals["chainuq"]))
    mac("vSilentShare", f"{100 * np.mean([S[f'{bb}/{d}']['silent_frac'] for bb in BBS for d in DSS]):.0f}", "{}")


def main():
    O = main_macros()
    I = accounting_table(O)
    pools_table()
    attribution_table()
    bigjudge_table()
    accounting_all_table(O)
    pools_bb_table()
    ablation_table()
    grid_table()
    for fn in (response_macros, decomposition_macros, bound_macros, judge_effort_macros, subtask_macros, distill_macros):
        fn()
    for fn in (comparison_macros, unseen_macros, think_macros):
        fn()
    route = routing()
    accepted_figure()
    teaser(I, route)
    pct = {"vRouteHalf", "vRouteRand", "vAccShare", "vSilent", "vSilentShare", "vRouteCross"}  # followed by \%: no \xspace
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}" + ("}" if k in pct else r"\xspace}") for k, v in sorted(macros.items()))
    (PAPER_DIR / "table" / "information_numbers.tex").write_text("% generated by chainuq.reports.information\n" + tex + "\n")
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
