"""Shared evaluation logic: evidence assembly per dataset cell, scoring, and aggregation over seeds.

Protocol: every calibrator and decision threshold is fit on the validation split
of the evaluated dataset and applied once to the evaluated split (``test`` or the
fresh held-out ``test2``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from chainuq import calibration as C
from chainuq import evidence as EV
from chainuq import metrics as M
from chainuq import registry
from chainuq.artifacts import index_by_uid, pred_path, read_jsonl, root
from chainuq.samples import load_judge_task, load_samples, load_step_labels, verdict_value

FEATURE_KEYS = ["tok", "ll", "top1", "margin", "ent"]


def verdict_score(rec: dict) -> float:
    """Judge-Final score: the chain verdict, with P(true) as a tie-breaker (the verbalized probability is coarse)."""
    v = verdict_value(rec)
    return np.nan if v is None else v + 1e-3 * (rec.get("p_true") or 0.0)


def available_methods(backbone: str) -> Dict[str, List[int]]:
    """method -> seeds with predictions for ``backbone``."""
    out = {}
    preds = root() / "preds"
    for mdir in sorted(preds.iterdir()) if preds.exists() else []:
        seeds = sorted(int(p.name[4:]) for p in (mdir / backbone).glob("seed*")
                       if p.is_dir() and p.name[4:].isdigit() and any(p.rglob("*.jsonl")))
        if seeds:
            out[mdir.name] = seeds
    return out


def load_preds(method: str, backbone: str, seed: int, dataset: str, split: str) -> Optional[dict]:
    path = pred_path(method, backbone, seed, dataset, split)
    return index_by_uid(read_jsonl(path)) if path.exists() else None


class Cell:
    """Validation and evaluated split of one (backbone, dataset), with cached evidence building blocks."""

    def __init__(self, backbone: str, dataset: str, split: str, hidden_dim: int):
        self.backbone, self.dataset, self.split, self.hidden_dim = backbone, dataset, split, hidden_dim
        self.val = load_samples(backbone, dataset, "validation", feature_keys=FEATURE_KEYS)
        self.test = load_samples(backbone, dataset, split, feature_keys=FEATURE_KEYS)
        self._static, self._cache = {}, {}

    def samples(self, split: str):
        if split == "validation":
            return self.val
        assert split == self.split, f"cell holds {self.split!r}, not {split!r}"
        return self.test

    def labels(self, split: str) -> np.ndarray:
        return np.asarray([s.y for s in self.samples(split)])

    def scores(self, split: str, preds: dict) -> np.ndarray:
        return np.asarray([preds[s.uid]["logit"] if s.uid in preds else np.nan for s in self.samples(split)],
                          dtype=np.float64)

    def span_features(self, split: str) -> np.ndarray:
        if ("span", split) not in self._cache:
            self._cache["span", split] = EV.matrix([EV.span_features(steps=s.steps, conclusion=s.conclusion, feat=s.feat)
                                                    for s in self.samples(split)], EV.SPAN_FEATURE_NAMES)
        return self._cache["span", split]

    def verdict(self, split: str) -> np.ndarray:
        """Chain verdict of the main judge (answer-blind), NaN where it failed."""
        if ("verdict", split) not in self._cache:
            rec = load_judge_task(registry.MAIN_JUDGE, "final_blind", self.backbone, self.dataset, split)
            self._cache["verdict", split] = np.asarray([verdict_score(rec.get(s.uid, {})) for s in self.samples(split)],
                                                       dtype=np.float64)
        return self._cache["verdict", split]

    def _static_features(self, s):
        if s.uid not in self._static:
            self._static[s.uid] = EV.static_features(steps=s.steps, conclusion=s.conclusion, n_hops=s.gen["n_hops"],
                                                     feat=s.feat, hidden_dim=self.hidden_dim)
        return self._static[s.uid]

    def u_matrix(self, split: str, preds: dict, step_labels: Optional[dict] = None) -> np.ndarray:
        """u(r) of every response; ``step_labels`` (uid -> labels) replaces the main judge's step judgments."""
        rows = []
        for s in self.samples(split):
            pr = preds[s.uid]
            labels = s.step_labels if step_labels is None else step_labels.get(s.uid)
            rows.append(EV.sample_features(logit=pr["logit"], step_logits=pr.get("step_logits"), steps=s.steps,
                                           conclusion=s.conclusion, n_hops=s.gen["n_hops"], feat=s.feat,
                                           hidden_dim=self.hidden_dim, step_labels=labels, static=self._static_features(s)))
        return EV.matrix(rows)

    def with_predicted_verdict(self, X: np.ndarray, split: str, preds: dict) -> np.ndarray:
        """Append the head's predicted chain verdict (PCS heads) to judge-free evidence."""
        if not preds or "verdict_hat" not in next(iter(preds.values())):
            return X
        return np.column_stack([X, [preds[s.uid].get("verdict_hat", np.nan) for s in self.samples(split)]])


def full_evidence(U, verdict, span):
    """Evidence of ChainUQ: u(r), the judge's chain verdict, and span token features."""
    return np.column_stack([U, verdict, span])


def judge_free_evidence(U, verdict, span):
    """Evidence of ChainUQ-A: u(r) without the judge's step judgments, and span token features."""
    return np.column_stack([U[:, EV.JUDGE_FREE_COLUMNS], span])


EVIDENCE = {"B": full_evidence, "A": judge_free_evidence}


class HeadEvidence:
    """Head scores and evidence matrices of one (head, seed) on the validation and evaluated split of a cell."""

    def __init__(self, cell: Cell, preds_val: dict, preds_test: dict, use_predicted_verdict: bool = True,
                 step_labels_val: Optional[dict] = None, step_labels_test: Optional[dict] = None):
        sp = cell.split
        self.sv, self.st = cell.scores("validation", preds_val), cell.scores(sp, preds_test)
        self.Uv = cell.u_matrix("validation", preds_val, step_labels_val)
        self.Ut = cell.u_matrix(sp, preds_test, step_labels_test)
        self.Ev, self.Et = cell.span_features("validation"), cell.span_features(sp)
        if use_predicted_verdict:
            self.Ev = cell.with_predicted_verdict(self.Ev, "validation", preds_val)
            self.Et = cell.with_predicted_verdict(self.Et, sp, preds_test)
        self.jv, self.jt = cell.verdict("validation"), cell.verdict(sp)

    def matrices(self, tier: str):
        build = EVIDENCE[tier]
        return build(self.Uv, self.jv, self.Ev), build(self.Ut, self.jt, self.Et)


def eval_scores(y_val, s_val, y_te, s_te, calibrated_te=None) -> dict:
    """Discrimination on the evaluated split (accuracy threshold from validation), plus calibration of
    ``calibrated_te`` if given."""
    ok_v, ok_t = np.isfinite(s_val), np.isfinite(s_te)
    out = M.discrimination(y_te[ok_t], s_te[ok_t], M.best_threshold(y_val[ok_v], s_val[ok_v]))
    out["n_test"] = int(ok_t.sum())
    if calibrated_te is not None:
        out.update(M.calibration(y_te[ok_t], calibrated_te[ok_t]))
    return out


def fit_eval(cls, y_val, s_val, X_val, y_te, s_te, X_te):
    """Fit a reasoning-aware calibrator on validation and score its held-out probabilities."""
    m = cls().fit(s_val, X_val, y_val)
    p_val, p_te = m.predict(s_val, X_val), m.predict(s_te, X_te)
    return eval_scores(y_val, p_val, y_te, p_te, p_te), m, p_te


def scalar_calibrated(name, y_val, s_val, s_te) -> np.ndarray:
    ok = np.isfinite(s_val)
    m = C.SCALAR[name]().fit(s_val[ok], y_val[ok])
    out = np.full_like(s_te, np.nan)
    good = np.isfinite(s_te)
    out[good] = m.predict(s_te[good])
    return out


def agg(rows: List[dict]) -> dict:
    """Mean/std over seeds for a list of metric dicts."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows])),
                "n_seeds": len(rows)} for k in keys}


def paired_bootstrap_auroc(y, scores_a, scores_b, n_boot=1000, seed=2026) -> dict:
    """95% interval of mean-over-seeds AUROC(A) - AUROC(B). ``scores_*``: [n_seeds, n]."""
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(np.mean([M.auroc(y[idx], s[idx]) for s in scores_a]) - np.mean([M.auroc(y[idx], s[idx]) for s in scores_b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def judge_step_labels(judge: str, backbone: str, dataset: str, split: str, task: str = "steps") -> dict:
    """Step judgments of a verifier (``self-<backbone>`` judges exist only for their own backbone)."""
    if judge.startswith("self-") and judge != f"self-{backbone}":
        return {}
    return load_step_labels(judge, backbone, dataset, split, task)
