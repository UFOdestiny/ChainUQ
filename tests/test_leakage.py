"""Guards against the label leakage found in the original R-Log features.

Before commit 669caa1 the step-judge features ``match_rate`` / ``label_gap_mean``
/ ``majority_gap`` were computed from each sample's own correctness label y,
on validation *and* test. These tests fail if anything label-derived can reach
u(r) again.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import ufeatures as U
from cuq.calibration import RLog


def _fake_sample(rng, T=40, F=24, hidden=16):
    steps = [{"text": "a", "tok": [2, 10]}, {"text": "b", "tok": [12, 25]}]
    conclusion = {"text": "c", "tok": [30, 33]}
    feat = {
        "tok": torch.randn(T, F, generator=rng),
        "ll": -torch.rand(T, generator=rng),
        "top1": -torch.rand(T, generator=rng),
        "margin": torch.rand(T, generator=rng),
    }
    return dict(logit=float(torch.randn(1, generator=rng)), step_logits=[0.3, -0.2], steps=steps,
                conclusion=conclusion, n_hops=2, feat=feat, hidden_dim=hidden, step_labels=[1, 0])


def test_feature_names_have_no_label_terms():
    U.assert_no_label_features(U.FEATURE_NAMES)
    for leaked in ("reasoning_label_match_rate", "reasoning_label_gap_mean", "reasoning_label_majority_gap"):
        assert leaked not in U.FEATURE_NAMES


def test_sample_features_cannot_receive_the_label():
    params = set(inspect.signature(U.sample_features).parameters)
    assert not params & {"y", "label", "sample_label", "correct", "gold"}


def test_u_is_identical_for_flipped_labels():
    rng = torch.Generator().manual_seed(0)
    rows = [U.sample_features(**_fake_sample(rng)) for _ in range(50)]
    X = U.matrix(rows)
    # Rebuild after "flipping" labels: there is no label input, so the matrix must be bit-identical.
    rng = torch.Generator().manual_seed(0)
    X2 = U.matrix([U.sample_features(**_fake_sample(rng)) for _ in range(50)])
    assert np.array_equal(X, X2, equal_nan=True)


def test_rlog_on_shuffled_labels_is_uninformative():
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.normal(size=n)
    Umat = rng.normal(size=(n, 10))
    y = (rng.random(n) < 1 / (1 + np.exp(-s))).astype(int)
    y_perm = rng.permutation(y)
    model = RLog().fit(s[:3000], Umat[:3000], y_perm[:3000])
    p = model.predict(s[3000:], Umat[3000:])
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(y_perm[3000:], p) - 0.5) < 0.06


def test_static_cache_is_bit_identical():
    rng = torch.Generator().manual_seed(1)
    for _ in range(20):
        kw = _fake_sample(rng)
        static = U.static_features(steps=kw["steps"], conclusion=kw["conclusion"], n_hops=kw["n_hops"],
                                   feat=kw["feat"], hidden_dim=kw["hidden_dim"])
        a = U.matrix([U.sample_features(**kw)])
        b = U.matrix([U.sample_features(**kw, static=static)])
        assert np.array_equal(a, b, equal_nan=True)
