"""The evidence vector u(r) must not be able to see the correctness label."""

import inspect

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from chainuq import evidence as EV
from chainuq.calibration import RLogCore


def _fake_sample(rng, T=40, F=24, hidden=16):
    feat = {"tok": torch.randn(T, F, generator=rng), "ll": -torch.rand(T, generator=rng),
            "top1": -torch.rand(T, generator=rng), "margin": torch.rand(T, generator=rng)}
    return dict(logit=float(torch.randn(1, generator=rng)), step_logits=[0.3, -0.2],
                steps=[{"text": "a", "tok": [2, 10]}, {"text": "b", "tok": [12, 25]}], conclusion={"text": "c", "tok": [30, 33]},
                n_hops=2, feat=feat, hidden_dim=hidden, step_labels=[1, 0])


def test_feature_names_have_no_label_terms():
    EV.assert_no_label_features(EV.FEATURE_NAMES + EV.SPAN_FEATURE_NAMES)


def test_sample_features_cannot_receive_the_label():
    params = set(inspect.signature(EV.sample_features).parameters)
    assert not params & {"y", "label", "sample_label", "correct", "gold"}


def test_judge_free_columns_exclude_step_judgments():
    assert not set(EV.JUDGE_FREE_COLUMNS) & set(EV.JUDGE_COLUMNS)
    assert len(EV.JUDGE_FREE_COLUMNS) + len(EV.JUDGE_COLUMNS) == len(EV.FEATURE_NAMES)


def test_static_cache_is_bit_identical():
    rng = torch.Generator().manual_seed(1)
    for _ in range(20):
        kw = _fake_sample(rng)
        static = EV.static_features(steps=kw["steps"], conclusion=kw["conclusion"], n_hops=kw["n_hops"],
                                    feat=kw["feat"], hidden_dim=kw["hidden_dim"])
        assert np.array_equal(EV.matrix([EV.sample_features(**kw)]), EV.matrix([EV.sample_features(**kw, static=static)]),
                              equal_nan=True)


def test_missing_step_judgments_are_nan():
    kw = _fake_sample(torch.Generator().manual_seed(0))
    row = EV.sample_features(**dict(kw, step_labels=None))
    assert all(np.isnan(row[n]) for n in EV.GROUPS["u_judge"][:-1]) and row["reasoning_label_valid_count"] == 0.0


def test_calibrator_on_shuffled_labels_is_uninformative():
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.normal(size=n)
    U = rng.normal(size=(n, 10))
    y = rng.permutation((rng.random(n) < 1 / (1 + np.exp(-s))).astype(int))
    p = RLogCore().fit(s[:3000], U[:3000], y[:3000]).predict(s[3000:], U[3000:])
    assert abs(roc_auc_score(y[3000:], p) - 0.5) < 0.06
