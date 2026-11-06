"""Information-decomposition tables, numerical summaries, Figure 1, and the accepted-responses figure of the paper.

Inputs (all fit on validation and scored on test unless stated), under ``CHAINUQ_ANALYSIS``:
  runs/<tag>.jsonl                            Stage-2 variant runs (``analysis.variants``)
  information/decomposition_<head>[_*].json   information decomposition (``analysis.information``)
  information/routing_<head>.json             information-gain routing (``analysis.routing``)
  information/{babi_subtasks,unseen_domain}.json, responses.json    (``analysis.subtasks``, ``.unseen``, ``.responses``)
and ``results/<backbone>/eval*.json`` (``pipeline.evaluate``, ``analysis.fusion``).

Writes ``table/{decomposition,accounting_all,pools,pools_bb,attribution,bigjudge,ablation,grid}.tex``
and ``fig/{teaser_a,teaser_b,teaser_c,accepted}.pdf`` under ``CHAINUQ_PAPER_DIR``.
Prints a numerical summary for manual verification and transcription. Manuscript numbers are
handwritten literals; do not generate numeric LaTeX macros or update prose automatically.

    python -m chainuq.reports.information
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.variants import EVIDENCE_A as EVA
from chainuq.analysis.variants import EVIDENCE_B as EVB
from chainuq.artifacts import results_dir
from chainuq.config import PAPER_DIR
from chainuq.registry import FINAL_HEAD, SOURCE_HEAD
from chainuq.analysis.information import BUDGETS
from chainuq.reports.style import BLUE, INK, LIGHT_BLUE, LIGHT_GRAY, ORANGE, TEAL, paper_plot_style

BBS = ["llama", "qwen8b", "ministral"]
DSS = ["hotpotqa", "musique", "strategyqa", "babi", "gsm8k", "math500"]
DSNAME = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "strategyqa": "StrategyQA", "babi": "bAbI", "gsm8k": "GSM8K", "math500": "MATH-500"}
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
        r"\textbf{Dataset} & $I_{A|J}$ & Accepted & $I_{J|A}$ & $\Delta$AUROC \\",
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
        r"\caption{Usable-information decomposition (nats; Eq.~\eqref{eq:accounting}), mean over backbones and 10 seeds. "
        r"$I_{A|J}$ and $I_{J|A}$: probe beyond judge and judge beyond probe. Accepted: the part of $I_{A|J}$ from responses with $v\geq0.5$. "
        r"$\Delta$AUROC: \m minus Judge-Final.}",
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
        ("None", "chainuq-wo-pcs"),
        ("Labeled source", SOURCE_HEAD),
        ("Unlabeled: steps", "chainuq-pool-steps"),
        ("Unlabeled: verdict level", "chainuq-pool-verdict"),
        (r"Unlabeled: both (\m)", FINAL_HEAD),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{l ccc cc}",
        r"\toprule",
        r"\textbf{Imitation} & $\epsilon_q$ & $I_{J|A}$ & $I_{A|J}$ & \m & \mJ \\",
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
        r"\end{tabular}",
        r"\caption{\textbf{Better imitation lowers the judge's residual and raises judge-free AUROC.} "
        r"Macro, 3 seeds; all rows include chain features and the second probe. Unlabeled: 6k questions per OOD dataset "
        r"added to the source. $\epsilon_q$: imitation loss, the held-out cross-entropy of the verdict level given $A$; "
        r"$I_{J|A}$ and $I_{A|J}$: judge beyond probe and probe beyond judge (Eq.~\eqref{eq:accounting}; all three in nats); last two columns: AUROC.}",
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
        (r"\multicolumn{4}{l}{\emph{Judge-free}} \\", None),
        ("Probe alone (isotonic)", {"saplma": "S@iso#saplma", "uhead": "S@iso#uhead", "ours": f"S@iso#{SOURCE_HEAD}"}),
        ("+ chain features", {"saplma": f"{CA}#saplma", "uhead": f"{CA}#uhead", "ours": f"{CA}#{SOURCE_HEAD}"}),
        (r"+ imitation, labeled source", {"uhead": f"{PA}#uhead-pcs", "ours": f"{PA}#{SOURCE_HEAD}"}),
        (r"+ imitation, unlabeled questions", {"uhead": f"{PA}#uhead-pcs-pool6k", "ours": f"{PA}#{FINAL_HEAD}"}),
        (r"+ second probe", {"uhead": f"{CHAINUQ_A}#uhead-pcs-pool6k", "ours": f"{CHAINUQ_A}#{FINAL_HEAD}"}),
        (r"\midrule \multicolumn{4}{l}{\emph{With the judge}} \\", None),
        (
            "Probe--judge fusion",
            {"saplma": "S+verdict+ptrue@lrsel#saplma", "uhead": "S+verdict+ptrue@lrsel#uhead", "ours": f"S+verdict+ptrue@lrsel#{SOURCE_HEAD}"},
        ),
        ("+ chain features", {"saplma": f"{CB}#saplma", "uhead": f"{CB}#uhead", "ours": f"{CB}#{SOURCE_HEAD}"}),
        (r"+ imitation, labeled source", {"uhead": f"{PB}#uhead-pcs", "ours": f"{PB}#{SOURCE_HEAD}"}),
        (r"+ imitation, unlabeled questions", {"uhead": f"{PB}#uhead-pcs-pool6k", "ours": f"{PB}#{FINAL_HEAD}"}),
        (r"+ second probe", {"uhead": f"{CHAINUQ_B}#uhead-pcs-pool6k", "ours": f"{CHAINUQ_B}#{FINAL_HEAD}"}),
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
        r"\caption{\textbf{Chain features and label-free imitation give the largest gains, for UHead as for our probe.} "
        r"Macro AUROC; each row adds to the one above, calibrated by ours from ``+ chain features'' on; our last rows are "
        r"\m and \mJ. 10 seeds (3 for UHead imitation and our first three rows).}",
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
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"\textbf{Judge} & Judge only & Fusion & \mJ & $I_{A|J}$ \\",
        r"\midrule",
    ]
    out = {}
    RQ, IQ = runs("cross_family_judge"), info_rows(T / f"decomposition_{FINAL_HEAD}_qwen3judge.json")
    RH, IH = runs("haiku_swap_haiku", "haiku_swap3_haiku"), info_rows(T / f"decomposition_{FINAL_HEAD}_haiku.json")
    keys = {"gpt-oss-20b": "gptoss", "gpt-oss-120b": "gptoss120b", "Qwen3-8B, thinking": "qwen3judge", "Claude Haiku 5.5 (API)": "haiku"}
    for name, R, I in (
        ("gpt-oss-20b", R20, I20),
        ("gpt-oss-120b", R120, I120),
        ("Qwen3-8B, thinking", RQ, IQ),
        ("Claude Haiku 5.5 (API)", RH, IH),
    ):
        jf = judge_auroc(keys[name])
        fus = max(macro(R, f"S+verdict+ptrue@lrsel#{h}") for h in ("saplma", "uhead"))
        b = macro(R, f"{CHAINUQ_B}#{FINAL_HEAD}")
        out[name] = (jf, fus, b)
        lines.append(f"{name} & {jf:.3f} & {fus:.3f} & {b:.3f} & {np.mean([I[k]['I_AgJ'] for k in I]):.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{The internal states add to every judge.} Macro AUROC with the judge swapped at inference; the probe "
        r"keeps its gpt-oss-20b training. Judge only: logistic regression on $J$ (Judge-Final in Table~\ref{tab:result} uses $v$ alone); Fusion: the stronger probe--judge fusion; $I_{A|J}$: probe beyond judge (nats; Eq.~\eqref{eq:accounting}).}",
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
    mac("vRouteFull", fus_)
    gap = np.abs(np.mean([r["auroc"]["eig"] for r in rows], 0) - np.mean([r["auroc"]["eig-mix"] for r in rows], 0)).max()
    mac("vRouteMixGap", gap)
    # probe + judge fusion under the same budget (``analysis.responses``): the stronger of the two fusions per
    # backbone and budget, routed by the probe's confidence; at 100% it equals the fusion of Table 1
    F = json.loads(RESPONSES.read_text())["fusion_routing"]
    R["fusion"] = np.mean(
        [np.max([np.mean([F[f"{bb}/{d}"][k]["conf"] for d in DSS], 0) for k in ("fusion-saplma", "fusion-uhead")], 0) for bb in BBS], 0
    )
    assert abs(R["fusion"][-1] - fus_) < 1e-9
    assert (R["eig"] > R["fusion"]).all()
    below = lambda v: str(int(round(100 * max(f for f, u in zip(BUDGETS, R["fusion"]) if u < v))))
    mac("vRouteFusHalf", R["fusion"][5])
    mac("vRouteFreeMatch", below(R["eig"][0]), "{}")  # fusion budget that ChainUQ exceeds without any judge call
    mac("vRouteThirtyMatch", below(R["eig"][3]), "{}")  # ... and with 30% of the calls
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


# Figure 1: three subfigures of the 7 in text width, each included at its natural size so that every label is set
# at the 10 pt body size; equal heights and bottom margins align the three axes.
PANEL_H, PANEL_BOTTOM, PANEL_TOP = 1.5, 0.37, 0.06
DSABBR = {"hotpotqa": "HQA", "musique": "MSQ", "strategyqa": "SQA", "babi": "bAbI", "gsm8k": "GSM", "math500": "MATH"}


def panel(width, left, right=0.08, height=PANEL_H, bottom=PANEL_BOTTOM, top=PANEL_TOP):
    """A one-axes figure of the given size in inches, with margins in inches."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, (height - bottom - top) / height])
    ax.tick_params(length=2.5, pad=2)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2
    return fig, ax


def save(fig, name):
    out = PAPER_DIR / "fig" / name
    fig.savefig(out)
    plt.close(fig)
    print(out)


def teaser_a(rel):
    """(a) accuracy per probe-confidence bin, split by the judge verdict; dashed ellipses mark where each source is
    overconfident: accepted answers with low probe confidence, and rejected answers with high probe confidence."""
    fig, a = panel(2.2, left=0.42)
    ylim = (0, 1.25)  # headroom above accuracy 1 holds the line and ellipse labels
    a.set(xlim=(0, 1), ylim=ylim, xticks=[0.5, 1], yticks=[0, 0.5, 1], xlabel="Probe confidence", ylabel="Accuracy")
    box = a.get_position()
    sx, sy = box.width * fig.get_figwidth(), box.height * fig.get_figheight() / (ylim[1] - ylim[0])  # inches per unit

    def ellipse(p, q, pad=0.09, minor=0.075):
        """Dashed ellipse (inches: round on the page) whose major axis runs through the points p and q."""
        (x0, y0), (x1, y1) = p, q
        dx, dy = (x1 - x0) * sx, (y1 - y0) * sy
        ang, major = np.arctan2(dy, dx), np.hypot(dx, dy) / 2 + pad
        t = np.linspace(0, 2 * np.pi, 200)
        u, v = major * np.cos(t), minor * np.sin(t)
        xs = (x0 + x1) / 2 + (u * np.cos(ang) - v * np.sin(ang)) / sx
        ys = (y0 + y1) / 2 + (u * np.sin(ang) + v * np.cos(ang)) / sy
        a.plot(xs, ys, color=INK, lw=0.7, ls=(0, (3, 2)))

    hx, hy = rel["high"]
    lx, ly = rel["low"]
    ellipse((hx[0], hy[0]), (hx[1], hy[1]))  # accepted, low probe confidence
    ellipse((lx[-2], ly[-2]), (lx[-1], ly[-1]))  # rejected, high probe confidence
    for name, color, marker, ls in (("low", ORANGE, "s", "--"), ("high", TEAL, "o", "-")):
        a.plot(*rel[name], color=color, marker=marker, ms=3.8, lw=1.25, ls=ls, markeredgecolor="white", markeredgewidth=0.4)
    arrow = dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6, shrinkA=1, shrinkB=1)
    a.text(0.99, 1.23, "Judge accepts", color=TEAL, va="top", ha="right")
    a.text(0.98, 0.03, "Judge rejects", color=ORANGE, va="bottom", ha="right")
    a.annotate("Judge overconf.", xy=(0.12, 0.41), xytext=(0.03, 0.92), va="top", arrowprops=arrow)
    a.annotate("Probe overconf.", xy=(0.69, 0.4), xytext=(0.99, 0.64), va="top", ha="right", arrowprops={**arrow, "relpos": (0.1, 0)})
    save(fig, "teaser_a.pdf")


def teaser_b(I):
    """(b) usable information of each source beyond the other, per dataset (horizontal bars: the labels stay at 10 pt),
    with one note per regime: the judge recomputes math, while QA errors read fluently."""
    fig, b = panel(2.35, left=0.47)
    y = np.arange(len(DSS))[::-1]
    pa = [np.mean([I[bb, d]["I_AgJ"] for bb in BBS]) for d in DSS]
    pj = [np.mean([I[bb, d]["I_JgA"] for bb in BBS]) for d in DSS]
    b.barh(y + 0.19, pa, height=0.36, color=TEAL, label="Probe beyond judge")
    b.barh(y - 0.19, pj, height=0.36, color=ORANGE, label="Judge beyond probe")
    b.set(yticks=y, yticklabels=[DSABBR[d] for d in DSS], ylim=(-0.6, len(DSS) - 0.4), xlabel="Usable information (nats)")
    b.set_xlim(0, max(max(pa), max(pj)) * 1.8)  # room for the legend and the notes to the right of the bars
    b.tick_params(axis="y", length=0)
    b.legend(loc="upper right", frameon=False, handlelength=0.8, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.2)
    right = b.get_xlim()[1] * 0.99
    row = {d: yy for d, yy in zip(DSS, y)}
    b.text(right, (row["strategyqa"] + row["babi"]) / 2, "QA: errors hide\nin fluent text", color=TEAL, ha="right", va="center", linespacing=0.95)
    b.text(right, (row["gsm8k"] + row["math500"]) / 2, "Math: judge\nrecomputes", color=ORANGE, ha="right", va="center", linespacing=0.95)
    save(fig, "teaser_b.pdf")


def teaser(I, route):
    paper_plot_style()
    teaser_a(reliability())
    teaser_b(I)
    # (c) AUROC per judge-call budget: ChainUQ (information-gain routing) and probe + judge fusion (confidence routing)
    fig, c = panel(2.25, left=0.47)
    xs = 100 * np.asarray(BUDGETS)
    c.plot(xs, route["fusion"], color=BLUE, lw=1.25, marker="^", ms=3.4, markeredgecolor="white", markeredgewidth=0.3)
    c.plot(xs, route["eig"], color=TEAL, lw=1.25, marker="D", ms=3.0, markeredgecolor="white", markeredgewidth=0.3)
    O = official()
    jf = np.mean([np.mean([O[bb][d]["judge-final"] for d in DSS]) for bb in BBS])
    c.plot([100], [jf], linestyle="none", marker="s", ms=4.2, color=ORANGE, markeredgecolor="white", markeredgewidth=0.4, clip_on=False, zorder=4)
    c.text(4, route["eig"][0] - 0.004, "JIP + Judge", color=TEAL, va="top")
    c.text(35, route["fusion"][4] - 0.02, "Probe–judge fusion", color=BLUE, va="top")
    c.text(95, jf, "Judge-Final", color=ORANGE, va="center", ha="right")
    c.set(xlim=(-4, 104), xticks=[0, 50, 100], ylim=(route["fusion"][0] - 0.008, route["eig"][-1] + 0.012), yticks=[0.8, 0.85, 0.9])
    c.set(xlabel="Judge calls (%)", ylabel="AUROC")
    save(fig, "teaser_c.pdf")


def accounting_all_table(O):
    I = info_rows(T / f"decomposition_{FINAL_HEAD}.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"\textbf{Dataset} & $I_{A|J}$ & Accepted & $I_{J|A}$ & $\Delta$AUROC \\",
    ]
    names = {"llama": "Llama", "qwen8b": "Qwen3", "ministral": "Ministral"}
    for bb in BBS:  # one block per backbone, headed by its name (a backbone column would not fit the column width)
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{names[bb]}}}}} \\"]
        for d in DSS:
            m = I[bb, d]
            lines.append(
                f"{DSNAME[d]} & {m['I_AgJ']:.3f} & {m['acc_part']:.3f} & {m['I_JgA']:.3f} & "
                f"{O[bb][d]['chainuq-a'] - O[bb][d]['judge-final']:+.3f} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Usable-information decomposition per backbone and dataset (nats; 10 seeds). Columns as in "
        r"Table~\ref{tab:decomposition}.}",
        r"\label{tab:accounting_all}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "accounting_all.tex").write_text("\n".join(lines) + "\n")


def pools_bb_table():
    R = runs("attribution")
    heads = [("None", "chainuq-wo-pcs"), ("Source only", SOURCE_HEAD), ("+ 2k unlab.", "chainuq-pool2k"), (r"+ 6k unlab.", FINAL_HEAD)]
    names = {"llama": "Llama", "qwen8b": "Qwen3", "ministral": "Ministral"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{ll ccc cc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Imitation} & $\epsilon_q$ & $I_{J|A}$ & $I_{A|J}$ & \m & \mJ \\",
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
        (r"\m / \mJ", f"{EVB}@rlogs#{H}", f"{EVA}@rlogs#{H}"),
        ("Calibrator without evidence selection", f"{EVB}@rlog#{H}", f"{EVA}@rlog#{H}"),
        ("Logistic regression, same inputs", f"{EVB}@lrsel#{H}", f"{EVA}@lrsel#{H}"),
        ("Boosted trees, same inputs", f"{EVB}@gbm#{H}", f"{EVA}@gbm#{H}"),
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
        r" & \multicolumn{3}{c}{\textbf{Judge-free}} & \multicolumn{3}{c}{\textbf{With judge}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"\textbf{Variant} & AUROC$\uparrow$ & NLL$\downarrow$ & ECE$\downarrow$ & AUROC$\uparrow$ & NLL$\downarrow$ & ECE$\downarrow$ \\",
        r"\midrule",
    ]
    for name, vb, va in rows:
        cells = []
        for v in (va, vb):
            for metric in ("auroc", "nll", "ece"):
                cells.append(f3(macro(R, v, metric=metric)) if v and v in R else "--")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Evidence and calibrator ablations (macro over datasets and backbones, 10 seeds). Full evidence adds "
        r"step-score summaries, whole-response token statistics, and (with the judge) the judge's step labels.}",
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
    heads = [("No imitation", "chainuq-wo-pcs"), ("Source only", SOURCE_HEAD), ("+ 6k unlabeled", FINAL_HEAD)]

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
        r"\caption{Judge-free AUROC when the probe and the second probe see 1--100\% of the HotpotQA labels; the "
        r"calibrator uses the target validation labels in every row (Llama and Qwen3, 3 seeds).}",
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
    """Errors the judge lets through: AUROC on the responses the judge accepts, per dataset, mean over the three
    backbones (judge score, the head's raw score, the probe + judge fusions, ChainUQ). One column wide at natural size."""
    paper_plot_style()
    S = json.loads(RESPONSES.read_text())["silent"]
    series = [
        ("Judge-Final", "judge", ORANGE),
        ("Our probe (raw)", "head", LIGHT_GRAY),
        ("SAPLMA + Judge", "fusion-saplma", BLUE),
        ("UHead + Judge", "fusion-uhead", LIGHT_BLUE),
        ("JIP + Judge", "chainuq", TEAL),
    ]
    vals = {k: [np.mean([S[f"{bb}/{d}"]["accepted"][k] for bb in BBS]) for d in DSS] for _, k, _ in series}
    width, height, left, bottom, top = 3.3, 1.9, 0.42, 0.22, 0.42
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, (width - left - 0.04) / width, (height - bottom - top) / height])
    x, w = np.arange(len(DSS)), 0.16
    for i, (name, k, c) in enumerate(series):
        ax.bar(x + (i - 2) * w, vals[k], width=w - 0.02, color=c, label=name)
    ax.set_xticks(x, [DSABBR[d] for d in DSS])
    ax.set(ylim=(0.5, 1.0), yticks=[0.5, 0.75, 1.0], ylabel="AUROC")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5, pad=2)
    ax.yaxis.labelpad = 2
    ax.legend(loc="lower center", bbox_to_anchor=(0.43, 1.01), ncol=3, frameon=False, handlelength=0.8, handletextpad=0.3,
              columnspacing=0.5, labelspacing=0.15, borderaxespad=0.0)
    save(fig, "accepted.pdf")
    mac("vAccJudge", np.mean(vals["judge"])), mac("vAccOurs", np.mean(vals["chainuq"]))
    mac("vAccSap", np.mean(vals["fusion-saplma"])), mac("vAccUh", np.mean(vals["fusion-uhead"]))
    mac("vSilentShare", f"{100 * np.mean([S[f'{bb}/{d}']['silent_frac'] for bb in BBS for d in DSS]):.0f}", "{}")


def quoted_table():
    """Comparisons quoted in Sections 5.4-5.6 that have no table of their own (values from the summaries above)."""
    m = macros
    rows = [
        (r"\ref{sec:rq3}", "Fusion with SAPLMA trained on labels / relabeled", f"{m['vSapFus']} / {m['vDistFus']}"),
        (r"\ref{sec:rq3}", "Leave-one-domain-out imitation above source-only imitation", f"{m['vLodoWins']} of 15 pairs"),
        (r"\ref{sec:rq3}", r"No target information: \mJ / Judge-Final (OOD)", f"{m['vUnseenB']} / {m['vZeroJ']}"),
        (r"\ref{sec:budget}", r"\mJ with 50\% of calls / fusion with all calls", f"{m['vRouteHalfAUC']} / {m['vRouteFull']}"),
        (r"\ref{sec:budget}", r"Gap recovered with 50\% of calls: EIG / random", f"{m['vRouteHalf']}\\% / {m['vRouteRand']}\\%"),
        (r"\ref{sec:budget}", r"Calls at which \mJ exceeds fusion with all calls", f"{m['vRouteCross']}\\%"),
        (r"\ref{sec:budget}", r"Fusion calls that \m exceeds without any call", f"{m['vRouteFreeMatch']}\\%"),
        (r"\ref{sec:robust}", r"Math, gpt-oss-120b high effort: \mJ / judge", f"{m['vHighB']} / {m['vHighJ']}"),
        (r"\ref{sec:robust}", r"Thinking traces: \mJ / fusion / judge", f"{m['vThinkB']} / {m['vThinkFus']} / {m['vThinkJ']}"),
        (r"\ref{sec:robust}", r"Thinking traces, judge reads them: \mJ / fusion", f"{m['vTraceB']} / {m['vTraceFus']}"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}c p{0.56\columnwidth} r@{}}",
        r"\toprule",
        r"\textbf{Sec.} & \textbf{Comparison} & \textbf{Value} \\",
        r"\midrule",
        *[f"{sec} & {name} & {val} \\\\" for sec, name, val in rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparisons quoted in Sections~\ref{sec:rq3}--\ref{sec:robust} (macro AUROC unless noted; fusion: the "
        r"stronger probe--judge fusion). Protocols: Appendices~\ref{sec:app_model} and~\ref{sec:app_pools}.}",
        r"\label{tab:quoted}",
        r"\end{table}",
    ]
    (PAPER_DIR / "table" / "quoted.tex").write_text("\n".join(lines) + "\n")


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
    quoted_table()
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    main()
