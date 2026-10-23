"""Calibrators return valid probabilities and use the evidence."""

import numpy as np
import pytest

from chainuq import calibration as C
from chainuq.metrics import auroc


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 4))
    z = 1.5 * U[:, 0] + rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    s = z + rng.normal(scale=2.0, size=n)  # a noisy head score
    return s, U, y


@pytest.mark.parametrize("name", sorted(C.SCALAR))
def test_scalar_calibrators(name):
    s, _, y = _data()
    p = C.SCALAR[name]().fit(s[:1000], y[:1000]).predict(s[1000:])
    assert p.shape == (500,) and np.all((p > 0) & (p < 1))


@pytest.mark.parametrize("cls", [C.RLog, C.RLogCore, C.BWH])
def test_reasoning_calibrators_use_evidence(cls):
    s, U, y = _data()
    m = cls().fit(s[:1000], U[:1000], y[:1000])
    p = m.predict(s[1000:], U[1000:])
    assert np.all((p > 0) & (p < 1))
    assert auroc(y[1000:], p) > auroc(y[1000:], s[1000:]) + 0.02


def test_rlog_choice_is_a_candidate():
    s, U, y = _data(seed=1)
    m = C.RLog().fit(s, U, y)
    assert m.choice in C.RLog.CANDIDATES and set(m.inner_losses) == set(C.RLog.CANDIDATES)


def test_cross_fit_handles_missing_evidence():
    s, U, y = _data(seed=2)
    U[::7, 1] = np.nan
    p = C.RLogCore(cross_fit=True, final="platt").fit(s, U, y).predict(s, U)
    assert np.all(np.isfinite(p))
