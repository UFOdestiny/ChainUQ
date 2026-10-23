"""Post-hoc calibrators (Stage 2). Fit on a validation split, applied once to a held-out split.

Scalar calibrators act on the head logit s only: Platt scaling (PS), temperature
scaling (TS), and isotonic regression (ISO).

``RLogCore`` is one reasoning-aware logistic calibrator (paper Sec. 3.3): the anchor
score is the scalar calibrator with the lowest 5-fold NLL; the reasoning-conditioned
score is an L2 logistic regression on [s; evidence]; the two are mixed with one weight
per quantile bin of the reasoning-conditioned score (21-point grid, binary
cross-entropy), followed by a final isotonic or Platt map. With ``cross_fit`` the
anchor and reasoning scores that the mixing weights and the final map are fit on are
out-of-fold predictions, which removes their in-sample optimism.

``RLog`` (R-Log, used by ChainUQ) chooses its capacity on the fitting split by inner
5-fold log-loss among: isotonic on s only, a strongly regularized logistic regression,
cross-fitted ``RLogCore`` at two regularization strengths (isotonic or Platt final
map), and shallow gradient boosting with a cross-fitted isotonic map.

``BWH`` (binwise hybrid) is a comparison calibrator: two logistic regressions on
[s; evidence] split at an anchor error risk of 0.2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
SEED = 2026


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def nll(y, p):
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _n_folds(y, folds):
    return max(2, min(folds, int(min(np.bincount(y, minlength=2)))))


def out_of_fold(fit_predict, n, y, folds=5, seed=SEED):
    """Out-of-fold predictions; ``fit_predict(train_idx, test_idx)`` returns predictions for ``test_idx``."""
    out = np.zeros(n)
    for tr, te in StratifiedKFold(_n_folds(y, folds), shuffle=True, random_state=seed).split(np.zeros(n), y):
        out[te] = fit_predict(tr, te)
    return out


class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s).reshape(-1, 1), y)
        return self

    def predict(self, s):
        return _clip(self.m.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1])


class Temperature:
    def fit(self, s, y):
        s = np.asarray(s, dtype=np.float64)
        res = minimize_scalar(lambda t: nll(y, 1 / (1 + np.exp(-s / t))), bounds=(0.05, 20.0), method="bounded")
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
    """Column-mean imputation of missing evidence (e.g. responses without a valid step judgment)."""

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
    """Standardized L2 logistic regression on an arbitrary feature matrix."""

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


def select_anchor(s, y, folds: int = 5, seed: int = SEED) -> str:
    """Scalar calibrator with the lowest cross-validated NLL."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {name: float(np.mean([nll(y[te], cls().fit(s[tr], y[tr]).predict(s[te])) for tr, te in skf.split(s, y)]))
              for name, cls in SCALAR.items()}
    return min(scores, key=scores.get)


class RLogCore:
    def __init__(self, C: float = 0.25, grid: int = 21, final: str = "iso", cross_fit: bool = False):
        self.C, self.grid, self.final, self.cross_fit = C, grid, final, cross_fit

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        X = np.column_stack([s, U])
        self.anchor_name = select_anchor(s, y)
        anchor_cls = SCALAR[self.anchor_name]
        self.anchor = anchor_cls().fit(s, y)
        self.re = LogReg(self.C).fit(X, y)
        if self.cross_fit:
            p_an = out_of_fold(lambda tr, te: anchor_cls().fit(s[tr], y[tr]).predict(s[te]), len(y), y)
            p_re = out_of_fold(lambda tr, te: LogReg(self.C).fit(X[tr], y[tr]).predict(X[te]), len(y), y)
        else:
            p_an, p_re = self.anchor.predict(s), self.re.predict(X)
        n_bins = 3 if len(y) >= 3000 else 2
        self.edges = np.unique(np.quantile(p_re, np.linspace(0, 1, n_bins + 1)[1:-1]))
        bins = np.digitize(p_re, self.edges)
        self.alphas, mixed = [], np.zeros_like(p_re)
        for b in range(len(self.edges) + 1):
            m = bins == b
            if not m.any():
                self.alphas.append(0.0)
                continue
            a = min((nll(y[m], a * p_re[m] + (1 - a) * p_an[m]), a) for a in np.linspace(0, 1, self.grid))[1]
            self.alphas.append(float(a))
            mixed[m] = a * p_re[m] + (1 - a) * p_an[m]
        self.map = Isotonic().fit(mixed, y) if self.final == "iso" else Platt().fit(_logit(mixed), y)
        return self

    def predict(self, s, U):
        s = np.asarray(s, dtype=np.float64)
        p_an, p_re = self.anchor.predict(s), self.re.predict(np.column_stack([s, U]))
        a = np.asarray(self.alphas)[np.digitize(p_re, self.edges)]
        mixed = a * p_re + (1 - a) * p_an
        return self.map.predict(mixed if self.final == "iso" else _logit(mixed))


class CrossFitGBM:
    """Shallow gradient boosting on [s; evidence]; the final isotonic map is fit on out-of-fold predictions."""

    @staticmethod
    def _gbm():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                              min_samples_leaf=40, random_state=SEED)

    def fit(self, X, y):
        self.imp = _Imputer().fit(X)
        Xi = self.imp.transform(X)
        oof = out_of_fold(lambda tr, te: self._gbm().fit(Xi[tr], y[tr]).predict_proba(Xi[te])[:, 1], len(y), y)
        self.m = self._gbm().fit(Xi, y)
        self.iso = Isotonic().fit(oof, y)
        return self

    def predict(self, X):
        return self.iso.predict(self.m.predict_proba(self.imp.transform(X))[:, 1])


class RLog:
    """R-Log: capacity chosen by inner cross-validated log-loss on the fitting split (see module docstring)."""

    CANDIDATES = ("iso_s", "lr_c0.01", "rlogcf_c0.05_platt", "rlogcf_c0.05", "rlogcf_c0.25", "gbmcf_iso")

    @staticmethod
    def _fit_one(name, s, U, y):
        if name == "iso_s":
            return Isotonic().fit(s, y)
        if name == "lr_c0.01":
            return LogReg(C=0.01).fit(np.column_stack([s, U]), y)
        if name == "gbmcf_iso":
            return CrossFitGBM().fit(np.column_stack([s, U]), y)
        C = float(name.split("_c")[1].split("_")[0])  # rlogcf_c<C>[_platt]
        return RLogCore(C=C, final="platt" if name.endswith("platt") else "iso", cross_fit=True).fit(s, U, y)

    @staticmethod
    def _predict_one(name, m, s, U):
        if name == "iso_s":
            return m.predict(s)
        if name.startswith("rlogcf"):
            return m.predict(s, U)
        return m.predict(np.column_stack([s, U]))

    def fit(self, s, U, y, folds: int = 5, seed: int = SEED):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        losses = {}
        for name in self.CANDIDATES:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(_n_folds(y, folds), shuffle=True, random_state=seed).split(s.reshape(-1, 1), y):
                oof[te] = self._predict_one(name, self._fit_one(name, s[tr], U[tr], y[tr]), s[te], U[te])
            losses[name] = nll(y, oof)
        self.choice = min(losses, key=losses.get)
        self.inner_losses = losses
        self.model = self._fit_one(self.choice, s, U, y)
        return self

    def predict(self, s, U):
        return _clip(self._predict_one(self.choice, self.model, np.asarray(s, dtype=np.float64), U))


class BWH:
    def __init__(self, threshold: float = 0.2, C: float = 1.0):
        self.threshold, self.C = threshold, C

    def _low_risk(self, s):
        return (1 - 1 / (1 + np.exp(-s))) < self.threshold

    def fit(self, s, U, y):
        s, y = np.asarray(s, dtype=np.float64), np.asarray(y)
        X = np.column_stack([s, U])
        lo = self._low_risk(s)
        self.models = {}
        for key, m in (("lo", lo), ("hi", ~lo)):
            fallback = m.sum() < 32 or len(np.unique(y[m])) < 2
            self.models[key] = LogReg(self.C).fit(X, y) if fallback else LogReg(self.C).fit(X[m], y[m])
        return self

    def predict(self, s, U):
        s = np.asarray(s, dtype=np.float64)
        X = np.column_stack([s, U])
        lo = self._low_risk(s)
        out = np.zeros(len(s))
        if lo.any():
            out[lo] = self.models["lo"].predict(X[lo])
        if (~lo).any():
            out[~lo] = self.models["hi"].predict(X[~lo])
        return _clip(out)
