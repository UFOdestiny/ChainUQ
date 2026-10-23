"""Post-hoc calibrators (fit on validation, applied to test). Positive class = correct.

Scalar calibrators act on the Stage-1 logit s only: Platt scaling (PS),
temperature scaling (TS), isotonic regression (ISO).

R-Log (paper Sec. 3.3): anchor score s_an = the scalar calibrator with the lowest
5-fold cross-validated NLL on validation; reasoning-conditioned score
s_re = L2 logistic regression on [s; u(r)]; quantile bins of s_re (3 bins if
n >= 3000 else 2); per-bin mixing weight alpha_b chosen on a 21-point grid by
binary cross-entropy (the paper's objective); final isotonic map.

BWH (binwise hybrid, Table 2 comparator): two logistic regressions on [s; u(r)]
split at anchor error-risk 0.2.

LR(u) / LR(s,u): plain logistic regressions used in the E2 decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


def _nll(y, p):
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s).reshape(-1, 1), y)
        return self

    def predict(self, s):
        return _clip(self.m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1])


class Temperature:
    def fit(self, s, y):
        s = np.asarray(s, dtype=np.float64)
        res = minimize_scalar(lambda t: _nll(y, 1 / (1 + np.exp(-s / t))), bounds=(0.05, 20.0), method="bounded")
        self.t = float(res.x)
        return self

    def predict(self, s):
        return _clip(1 / (1 + np.exp(-np.asarray(s, dtype=np.float64) / self.t)))


class Isotonic:
    def fit(self, s, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(s, dtype=np.float64), y)
        return self

    def predict(self, s):
        return _clip(self.m.predict(np.asarray(s, dtype=np.float64)))


SCALAR = {"ps": Platt, "ts": Temperature, "iso": Isotonic}


class _Imputer:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(self.mu, idx[1])
        return X


class LogReg:
    """Standardised L2 logistic regression on an arbitrary feature matrix."""

    def __init__(self, C: float = 0.25):
        self.C = C

    def fit(self, X, y):
        self.imp = _Imputer().fit(X)
        Xi = self.imp.transform(X)
        self.sc = StandardScaler().fit(Xi)
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc.transform(Xi), y)
        return self

    def predict(self, X):
        return _clip(self.m.predict_proba(self.sc.transform(self.imp.transform(X)))[:, 1])


def select_anchor(s, y, folds: int = 5, seed: int = 2026) -> str:
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {}
    for name, cls in SCALAR.items():
        losses = []
        for tr, te in skf.split(s, y):
            losses.append(_nll(y[te], cls().fit(s[tr], y[tr]).predict(s[te])))
        scores[name] = float(np.mean(losses))
    return min(scores, key=scores.get)


class RLog:
    def __init__(self, C: float = 0.25, grid: int = 21, final: str = "iso"):
        self.C, self.grid, self.final = C, grid, final

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        self.anchor_name = select_anchor(s, y)
        self.anchor = SCALAR[self.anchor_name]().fit(s, y)
        p_an = self.anchor.predict(s)
        self.re = LogReg(self.C).fit(np.column_stack([s, U]), y)
        p_re = self.re.predict(np.column_stack([s, U]))
        n_bins = 3 if len(y) >= 3000 else 2
        self.edges = np.unique(np.quantile(p_re, np.linspace(0, 1, n_bins + 1)[1:-1]))
        bins = np.digitize(p_re, self.edges)
        self.alphas = []
        mixed = np.zeros_like(p_re)
        for b in range(len(self.edges) + 1):
            m = bins == b
            if not m.any():
                self.alphas.append(0.0)
                continue
            losses = [(_nll(y[m], a * p_re[m] + (1 - a) * p_an[m]), a) for a in np.linspace(0, 1, self.grid)]
            a = min(losses)[1]
            self.alphas.append(float(a))
            mixed[m] = a * p_re[m] + (1 - a) * p_an[m]
        self.iso = (Isotonic() if self.final == "iso" else Platt()).fit(mixed if self.final == "iso" else _logit(mixed), y)
        return self

    def _mixed(self, s, U):
        p_an = self.anchor.predict(s)
        p_re = self.re.predict(np.column_stack([s, U]))
        bins = np.digitize(p_re, self.edges)
        a = np.asarray(self.alphas)[bins]
        return a * p_re + (1 - a) * p_an

    def predict(self, s, U):
        mixed = self._mixed(np.asarray(s, dtype=np.float64), U)
        return self.iso.predict(mixed if self.final == "iso" else _logit(mixed))


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


class RLogVariant(RLog):
    """R-Log internal ablations for E6: fixed alpha (no bins) and/or no final isotonic map."""

    def __init__(self, single_alpha: bool = False, final_iso: bool = True, **kw):
        super().__init__(**kw)
        self.single_alpha, self.final_iso = single_alpha, final_iso

    def fit(self, s, U, y):
        if self.single_alpha:
            s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
            self.anchor_name = select_anchor(s, y)
            self.anchor = SCALAR[self.anchor_name]().fit(s, y)
            self.re = LogReg(self.C).fit(np.column_stack([s, U]), y)
            p_an, p_re = self.anchor.predict(s), self.re.predict(np.column_stack([s, U]))
            a = min((_nll(y, a * p_re + (1 - a) * p_an), a) for a in np.linspace(0, 1, self.grid))[1]
            self.edges, self.alphas = np.asarray([]), [float(a)]
            mixed = a * p_re + (1 - a) * p_an
            self.iso = Isotonic().fit(mixed, y)
            return self
        return super().fit(s, U, y)

    def predict(self, s, U):
        mixed = self._mixed(np.asarray(s, dtype=np.float64), U)
        return self.iso.predict(mixed) if self.final_iso else _clip(mixed)


class BWH:
    def __init__(self, threshold: float = 0.2, C: float = 1.0):
        self.threshold, self.C = threshold, C

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        X = np.column_stack([s, U])
        risk = 1 - 1 / (1 + np.exp(-s))
        lo = risk < self.threshold
        self.models = {}
        for key, m in (("lo", lo), ("hi", ~lo)):
            if m.sum() < 32 or len(np.unique(y[m])) < 2:
                self.models[key] = LogReg(self.C).fit(X, y)
            else:
                self.models[key] = LogReg(self.C).fit(X[m], y[m])
        return self

    def predict(self, s, U):
        s = np.asarray(s, dtype=np.float64)
        X = np.column_stack([s, U])
        lo = (1 - 1 / (1 + np.exp(-s))) < self.threshold
        out = np.zeros(len(s))
        if lo.any():
            out[lo] = self.models["lo"].predict(X[lo])
        if (~lo).any():
            out[~lo] = self.models["hi"].predict(X[~lo])
        return _clip(out)


class _GBMIso:
    """Shallow gradient boosting on [s; u] followed by an isotonic map."""

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.imp = _Imputer().fit(X)
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                min_samples_leaf=40, random_state=2026)
        Xi = self.imp.transform(X)
        self.m.fit(Xi, y)
        self.iso = Isotonic().fit(self.m.predict_proba(Xi)[:, 1], y)
        return self

    def predict(self, X):
        return self.iso.predict(self.m.predict_proba(self.imp.transform(X))[:, 1])


class RLog2:
    """R-Log v2: reasoning-evidence calibrator whose capacity is chosen by inner cross-validation.

    Inputs are the Stage-1 anchor score s and the evidence matrix U (u(r), optionally extended
    with the chain-level judge verdict). Candidates: R-Log with strong (C=0.05) or default
    (C=0.25) regularisation, and a shallow GBM + isotonic map. The candidate with the lowest
    inner 5-fold log-loss on the fitting data is refit on all of it. Fitting data is always the
    validation split; test is only ever passed to ``predict``.
    """

    CANDIDATES = ("rlog_c0.05", "rlog_c0.25", "gbm_iso")

    def _make(self, name):
        if name.startswith("rlog"):
            return RLog(C=float(name.split("_c")[1]))
        return _GBMIso()

    def _fit_one(self, name, s, U, y):
        m = self._make(name)
        return m.fit(s, U, y) if name.startswith("rlog") else m.fit(np.column_stack([s, U]), y)

    def _predict_one(self, name, m, s, U):
        return m.predict(s, U) if name.startswith("rlog") else m.predict(np.column_stack([s, U]))

    def fit(self, s, U, y, folds: int = 5, seed: int = 2026):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        n_min = int(min(np.bincount(y, minlength=2)))
        k = max(2, min(folds, n_min))
        losses = {}
        for name in self.CANDIDATES:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(s.reshape(-1, 1), y):
                oof[te] = self._predict_one(name, self._fit_one(name, s[tr], U[tr], y[tr]), s[te], U[te])
            losses[name] = _nll(y, oof)
        self.choice = min(losses, key=losses.get)
        self.inner_losses = losses
        self.model = self._fit_one(self.choice, s, U, y)
        return self

    def predict(self, s, U):
        return _clip(self._predict_one(self.choice, self.model, np.asarray(s, dtype=np.float64), U))



class RLog2b(RLog2):
    """R-Log v2 with a wider capacity ladder for small validation sets (chosen by the same inner CV):
    anchor-only isotonic < strongly regularised logistic < R-Log (Platt or isotonic final map) < GBM."""

    CANDIDATES = ("iso_s", "lr_c0.01", "rlog_c0.05_platt", "rlog_c0.05", "rlog_c0.25", "gbm_iso")

    def _make(self, name):
        if name == "iso_s":
            return Isotonic()
        if name == "lr_c0.01":
            return LogReg(C=0.01)
        if name == "rlog_c0.05_platt":
            return RLog(C=0.05, final="platt")
        return super()._make(name)

    def _fit_one(self, name, s, U, y):
        if name == "iso_s":
            return Isotonic().fit(s, y)
        if name == "lr_c0.01":
            return LogReg(C=0.01).fit(np.column_stack([s, U]), y)
        if name == "rlog_c0.05_platt":
            return RLog(C=0.05, final="platt").fit(s, U, y)
        return super()._fit_one(name, s, U, y)

    def _predict_one(self, name, m, s, U):
        if name == "iso_s":
            return m.predict(s)
        if name == "lr_c0.01":
            return m.predict(np.column_stack([s, U]))
        if name == "rlog_c0.05_platt":
            return m.predict(s, U)
        return super()._predict_one(name, m, s, U)


def _oof(fit_predict, n, y, folds=5, seed=2026):
    """Out-of-fold predictions: fit_predict(train_idx, test_idx) -> preds for test_idx."""
    out = np.zeros(n)
    k = max(2, min(folds, int(min(np.bincount(y, minlength=2)))))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y):
        out[te] = fit_predict(tr, te)
    return out


class RLogCF(RLog):
    """R-Log with cross-fitted calibration steps.

    The anchor score, the reasoning-conditioned score, the per-bin mixing weights and the final
    map are all fit on OUT-OF-FOLD predictions on the fitting split; prediction uses models refit
    on the whole split. This removes the in-sample optimism that made the final isotonic map
    over-confident on held-out data.
    """

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        X = np.column_stack([s, U])
        self.anchor_name = select_anchor(s, y)
        an_cls = SCALAR[self.anchor_name]
        p_an = _oof(lambda tr, te: an_cls().fit(s[tr], y[tr]).predict(s[te]), len(y), y)
        p_re = _oof(lambda tr, te: LogReg(self.C).fit(X[tr], y[tr]).predict(X[te]), len(y), y)
        self.anchor = an_cls().fit(s, y)
        self.re = LogReg(self.C).fit(X, y)
        n_bins = 3 if len(y) >= 3000 else 2
        self.edges = np.unique(np.quantile(p_re, np.linspace(0, 1, n_bins + 1)[1:-1]))
        bins = np.digitize(p_re, self.edges)
        self.alphas = []
        mixed = np.zeros_like(p_re)
        for b in range(len(self.edges) + 1):
            m = bins == b
            if not m.any():
                self.alphas.append(0.0)
                continue
            a = min((_nll(y[m], a * p_re[m] + (1 - a) * p_an[m]), a) for a in np.linspace(0, 1, self.grid))[1]
            self.alphas.append(float(a))
            mixed[m] = a * p_re[m] + (1 - a) * p_an[m]
        self.iso = (Isotonic() if self.final == "iso" else Platt()).fit(mixed if self.final == "iso" else _logit(mixed), y)
        return self


class _GBMIsoCF(_GBMIso):
    """GBM whose final isotonic map is fit on out-of-fold GBM predictions."""

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                    min_samples_leaf=40, random_state=2026)
        self.imp = _Imputer().fit(X)
        Xi = self.imp.transform(X)
        oof = _oof(lambda tr, te: mk().fit(Xi[tr], y[tr]).predict_proba(Xi[te])[:, 1], len(y), y)
        self.m = mk().fit(Xi, y)
        self.iso = Isotonic().fit(oof, y)
        return self


class RLog2c(RLog2b):
    """R-Log v2c: the v2b capacity ladder with every calibration step cross-fitted."""

    CANDIDATES = ("iso_s", "lr_c0.01", "rlogcf_c0.05_platt", "rlogcf_c0.05", "rlogcf_c0.25", "gbmcf_iso")

    def _fit_one(self, name, s, U, y):
        if name.startswith("rlogcf"):
            C_ = float(name.split("_c")[1].split("_")[0])
            final = "platt" if name.endswith("platt") else "iso"
            return RLogCF(C=C_, final=final).fit(s, U, y)
        if name == "gbmcf_iso":
            return _GBMIsoCF().fit(np.column_stack([s, U]), y)
        return super()._fit_one(name, s, U, y)

    def _predict_one(self, name, m, s, U):
        if name.startswith("rlogcf"):
            return m.predict(s, U)
        if name == "gbmcf_iso":
            return m.predict(np.column_stack([s, U]))
        return super()._predict_one(name, m, s, U)


class _GBMPlattCF(_GBMIsoCF):
    """GBM with a Platt (logistic) final map fit on out-of-fold predictions."""

    def fit(self, X, y):
        super().fit(X, y)
        Xi = self.imp.transform(X)
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                    min_samples_leaf=40, random_state=2026)
        oof = _oof(lambda tr, te: mk().fit(Xi[tr], y[tr]).predict_proba(Xi[te])[:, 1], len(y), y)
        self.iso = None
        self.platt = Platt().fit(_logit(oof), y)
        return self

    def predict(self, X):
        return self.platt.predict(_logit(self.m.predict_proba(self.imp.transform(X))[:, 1]))


class RLog2d(RLog2c):
    """Capacity ladder whose every final map is a smooth Platt map (fit out-of-fold)."""

    CANDIDATES = ("platt_s", "lr_c0.01", "rlogcf_c0.05_platt", "rlogcf_c0.25_platt", "gbmcf_platt")

    def _fit_one(self, name, s, U, y):
        if name == "platt_s":
            return Platt().fit(s, y)
        if name == "rlogcf_c0.25_platt":
            return RLogCF(C=0.25, final="platt").fit(s, U, y)
        if name == "gbmcf_platt":
            return _GBMPlattCF().fit(np.column_stack([s, U]), y)
        return super()._fit_one(name, s, U, y)

    def _predict_one(self, name, m, s, U):
        if name == "platt_s":
            return m.predict(s)
        if name == "rlogcf_c0.25_platt":
            return m.predict(s, U)
        if name == "gbmcf_platt":
            return m.predict(np.column_stack([s, U]))
        return super()._predict_one(name, m, s, U)


class Smoothed:
    """Wrap a calibrator and add one Platt map fit on its out-of-fold outputs."""

    def __init__(self, base_cls, folds: int = 3):
        self.base_cls, self.folds = base_cls, folds

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        oof = _oof(lambda tr, te: self.base_cls().fit(s[tr], U[tr], y[tr]).predict(s[te], U[te]), len(y), y, folds=self.folds)
        self.base = self.base_cls().fit(s, U, y)
        self.platt = Platt().fit(_logit(oof), y)
        return self

    def predict(self, s, U):
        return self.platt.predict(_logit(self.base.predict(s, U)))
