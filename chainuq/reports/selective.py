"""Selective-risk and second-labeler tables of the paper.

Reads ``responses.json`` and the per-response predictions under ``CHAINUQ_ANALYSIS`` (``analysis.responses``,
``analysis.predictions``) and ``results/<backbone>/eval*.json``. Writes ``table/{selective_main,selective,relabel}.tex``
under ``CHAINUQ_PAPER_DIR``.

    python -m chainuq.reports.selective
"""

from __future__ import annotations

import json

import numpy as np

from chainuq.analysis import variants as D
from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.fusion import FUSION
from chainuq.analysis.predictions import out_path
from chainuq.analysis.responses import aurc, baseline_scores
from chainuq.analysis.responses import load as load_preds
from chainuq.config import PAPER_DIR
from chainuq.reports.results import BACKBONES, DATASETS, load, wrap_table



BB_SHORT = {"llama": "Llama", "qwen8b": "Qwen3", "ministral": "Ministral"}


def wrap1(lines, cols, caption, label, sep="3pt"):
    """Single-column float (``results.wrap_table`` always widens floats to both columns)."""
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            rf"\setlength{{\tabcolsep}}{{{sep}}}",
            rf"\begin{{tabular}}{{{cols}}}",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def f3(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.3f}"


def bold_best(vals, lower):
    ok = [round(v, 3) for v in vals if v is not None and np.isfinite(v)]
    best = min(ok) if lower else max(ok)
    return [rf"\textbf{{{f3(v)}}}" if v is not None and np.isfinite(v) and round(v, 3) == best else f3(v) for v in vals]


SEL_ROWS = [
    ("judge-final", "Judge-Final"),
    ("uhead", "UHead"),
    ("saplma", "SAPLMA"),
    ("selfcheck", "SelfCheckGPT$^*$"),
    ("fusion-saplma", "SAPLMA + Judge"),
    ("fusion-uhead", "UHead + Judge"),
    ("chainuq_a", r"\m"),
    ("chainuq", r"\mJ"),
    ("oracle", "Oracle"),
]


def selective_table(R):
    vals = {}
    for bb, _ in BACKBONES:
        for ds, _ in DATASETS:
            cell = R["aurc"][f"{bb}/{ds}"]
            for k, _ in SEL_ROWS:
                if k.startswith("fusion-"):
                    z = np.load(out_path(FUSION[k], bb, ds, "test"))
                    v = float(np.mean([aurc(z["y"], p) for p in z["p"]]))
                else:
                    v = cell.get(k)
                vals[(k, bb, ds)] = v
    groups = [(bb, name) for bb, name in BACKBONES]
    cols = "l" + " ccc" * len(groups)
    lines = [
        r"\toprule",
        " & ".join([""] + [rf"\multicolumn{{3}}{{c}}{{\textbf{{{BB_SHORT[bb]}}}}}" for bb, _ in groups]) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(groups))),
        r"\textbf{Method} & ID & OOD & Macro & ID & OOD & Macro & ID & OOD & Macro \\",
        r"\midrule",
    ]
    table = {}
    for k, _ in SEL_ROWS:
        for bb, _ in groups:
            ood = [vals[(k, bb, ds)] for ds, _ in DATASETS if ds != "hotpotqa"]
            table[(k, bb)] = [vals[(k, bb, "hotpotqa")], float(np.mean(ood)), float(np.mean([vals[(k, bb, ds)] for ds, _ in DATASETS]))]
    marks = {}
    for bb, _ in groups:
        for i in range(3):
            col = [table[(k, bb)][i] for k, _ in SEL_ROWS if k != "oracle"]
            for (k, _), s in zip([r for r in SEL_ROWS if r[0] != "oracle"], bold_best(col, True)):
                marks[(k, bb, i)] = s
    for k, name in SEL_ROWS:
        if k == "oracle":
            lines.append(r"\midrule")
        cells = [marks.get((k, bb, i), f3(table[(k, bb)][i])) for bb, _ in groups for i in range(3)]
        shade = r"\rowcolor{gray!20} " if k in ("chainuq", "chainuq_a") else ""
        lines.append(f"{shade}{name} & " + " & ".join(cells) + r" \\")
    cap = (
        r"AURC$\downarrow$ per backbone (mean over seeds). OOD: mean over the five OOD datasets; Macro: over all six. "
        r"Oracle ranks every correct answer first. $^*$First 1,000 test questions of each dataset. Bold: best non-oracle method."
    )
    return wrap_table(lines, cols, cap, "tab:selective", star=True, sep="4pt")


RELABEL_ROWS = [
    ("selfcheck", "tier_a", "SelfCheckGPT"),
    ("saplma", "tier_a", "SAPLMA"),
    ("uhead", "tier_a", "UHead"),
    ("judge-final", "tier_b", "Judge-Final"),
    ("fusion-saplma", "tier_b", "SAPLMA + Judge"),
    ("fusion-uhead", "tier_b", "UHead + Judge"),
    ("chainuq-a", "tier_a", r"\m"),
    ("chainuq", "tier_b", r"\mJ"),
]


JUDGED = ["hotpotqa", "musique", "babi"]


def relabel_table():
    cols = "l" + " cc" * len(BACKBONES)
    lines = [
        r"\toprule",
        " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{{BB_SHORT[bb]}}}}}" for bb, _ in BACKBONES]) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BACKBONES))),
        r"\textbf{Method} & " + " & ".join(["gpt-oss", "Qwen3-4B"] * len(BACKBONES)) + r" \\",
        r"\midrule",
    ]
    vals = {}
    agree = {}
    for bb, _ in BACKBONES:
        R = load(bb)["datasets"]
        agree[bb] = float(np.mean([R[ds]["label_robustness"]["label_agreement"] for ds in JUDGED]))
        R2 = json.loads((ANALYSIS / "responses.json").read_text())["relabel"]
        for k, tier, _ in RELABEL_ROWS:
            vals[(k, bb, 0)] = float(np.mean([R[ds][tier][k]["auroc"]["mean"] for ds in JUDGED]))
            if k.startswith("fusion-"):
                vals[(k, bb, 1)] = float(np.mean([R2[f"{bb}/{ds}"][k] for ds in JUDGED]))
            else:
                vals[(k, bb, 1)] = float(np.mean([R[ds]["label_robustness"]["methods"][k]["second_judge_labels"]["auroc"]["mean"] for ds in JUDGED]))
    marks = {}
    for bb, _ in BACKBONES:
        for i in range(2):
            for (k, _, _), s in zip(RELABEL_ROWS, bold_best([vals[(k, bb, i)] for k, _, _ in RELABEL_ROWS], False)):
                marks[(k, bb, i)] = s
    for k, _, name in RELABEL_ROWS:
        shade = r"\rowcolor{gray!20} " if k.startswith("chainuq") else ""
        lines.append(f"{shade}{name} & " + " & ".join(marks[(k, bb, i)] for bb, _ in BACKBONES for i in range(2)) + r" \\")
    cap = (
        "AUROC against the labels of the main judge (gpt-oss) and of a second labeler (Qwen3-4B), mean over HotpotQA, "
        "MuSiQue, and bAbI. The labelers agree on "
        + ", ".join(f"{100 * agree[bb]:.1f}\\%" for bb, _ in BACKBONES)
        + r" of the responses (Llama, Qwen3, Ministral). Calibrators use the main labels. Bold: best per column."
    )
    return wrap_table(lines, cols, cap, "tab:relabel", star=True, sep="4pt")


def err_at(y, conf, cov=0.8):
    ok = np.isfinite(conf)
    y, conf = np.asarray(y)[ok], np.asarray(conf)[ok]
    k = int(np.ceil(cov * len(y)))
    return float(1 - y[np.argsort(-conf, kind="stable")[:k]].mean())


MAIN_SEL = [
    ("judge-final", "Judge-Final"),
    ("uhead", "UHead"),
    ("fusion-saplma", "SAPLMA + Judge"),
    ("fusion-uhead", "UHead + Judge"),
    ("chainuq_a", r"\m"),
    ("chainuq", r"\mJ"),
]


def selective_main_table():
    """Macro AURC per backbone and its mean over the backbones."""
    src = {"chainuq": D.CHAINUQ, "chainuq_a": D.CHAINUQ_A, **FUSION}
    vals = {}
    for bb, _ in BACKBONES:
        for k, _ in MAIN_SEL:
            a, e = [], []
            for ds, _ in DATASETS:
                if k in src:
                    z = load_preds(src[k], bb, ds)
                    y, S = z["y"], z["p"]
                else:
                    z = load_preds(D.CHAINUQ, bb, ds)
                    y = z["y"]
                    S = baseline_scores(k, bb, ds, [str(u) for u in z["uids"]])
                a.append(np.mean([aurc(y, s) for s in S]))
                e.append(np.mean([err_at(y, s) for s in S]))
            vals[(k, bb)] = (float(np.mean(a)), float(np.mean(e)))
    for k, _ in MAIN_SEL:
        vals[(k, "mean")] = (float(np.mean([vals[(k, bb)][0] for bb, _ in BACKBONES])), None)
    groups = [bb for bb, _ in BACKBONES] + ["mean"]
    cols = "l" + " c" * len(groups)
    lines = [
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(rf"\textbf{{{BB_SHORT.get(g, 'Mean')}}}" for g in groups) + r" \\",
        r"\midrule",
    ]
    marks = {}
    for g in groups:
        for (k, _), s in zip(MAIN_SEL, bold_best([vals[(k, g)][0] for k, _ in MAIN_SEL], True)):
            marks[(k, g)] = s
    for k, name in MAIN_SEL:
        shade = r"\rowcolor{gray!20} " if k.startswith("chainuq") else ""
        lines.append(f"{shade}{name} & " + " & ".join(marks[(k, g)] for g in groups) + r" \\")
    cap = (
        r"\textbf{\m has a lower selective risk than the judge, and \mJ the lowest on every backbone.} "
        r"AURC$\downarrow$, mean over the six datasets."
    )
    return wrap1(lines, cols, cap, "tab:selective_main", sep="4pt")


def main():
    R = json.loads((ANALYSIS / "responses.json").read_text())
    out = PAPER_DIR / "table"
    (out / "selective_main.tex").write_text(selective_main_table())
    (out / "selective.tex").write_text(selective_table(R))
    (out / "relabel.tex").write_text(relabel_table())
    print(out)


if __name__ == "__main__":
    main()
