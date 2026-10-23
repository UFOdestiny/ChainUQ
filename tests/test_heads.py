"""Stage-1 heads: budget sizing, padding invariance, and the outputs used for PCS."""

import pytest
import torch

from chainuq import heads as H

DIMS = {"feature_dim": 48, "hidden_size": 32, "n_layers": 4}


def _items(n=5, T=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        t = int(torch.randint(10, T + 1, (1,), generator=g))
        s = int(torch.randint(1, t - 4, (1,), generator=g))
        feat = {"tok": torch.randn(t, DIMS["feature_dim"], generator=g),
                "layers_anchor": torch.randn(DIMS["n_layers"] + 1, DIMS["hidden_size"], generator=g),
                "layers_concl_last": torch.randn(DIMS["n_layers"] + 1, DIMS["hidden_size"], generator=g)}
        items.append({"uid": str(i), "y": i % 2, "feat": feat, "concl": (s, s + 3), "steps": [(0, s)],
                      "step_labels": [1], "verdict": 0.5})
    return items


@pytest.mark.parametrize("method", [m for m in H.HEADS if H.HEADS[m].arch in ("chainuq", "uhead", "uhead-pcs", "spanpool")])
def test_padding_invariance(method):
    from chainuq.training import collate
    torch.manual_seed(0)
    model = H.build(method, DIMS, 32).eval()
    items = _items()
    with torch.no_grad():
        batched = model(collate(items, "cpu"))
        single = torch.cat([model(collate([it], "cpu")) for it in items])
    torch.testing.assert_close(batched, single, rtol=1e-4, atol=1e-5)


def test_pcs_outputs():
    from chainuq.training import collate
    model = H.build("chainuq", DIMS, 32).eval()
    batch = collate(_items(), "cpu")
    logit, verdict = model.forward_all(batch)
    assert logit.shape == verdict.shape == (5,)
    torch.testing.assert_close(logit, model(batch))
    assert model.step_logits(batch).shape == (5,)


def test_ablations_drop_their_modules():
    assert H.build("chainuq-wo-pcs-token", DIMS, 32).token is None
    assert H.build("chainuq-wo-pcs-layer", DIMS, 32).layer_proj is None
    assert not hasattr(H.build("chainuq-wo-pcs", DIMS, 32), "verdict")
    assert not H.predicts_verdict("chainuq-wo-verdict") and H.predicts_verdict("chainuq")


def test_budget_width_of_the_final_head():
    dims = {"feature_dim": 4202, "hidden_size": 4096, "n_layers": 32}  # Llama-3.1-8B features
    width = H.size_to_budget("chainuq", dims)
    assert width == 112 and H.count_params(H.build("chainuq", dims, width)) <= H.PARAM_BUDGET
