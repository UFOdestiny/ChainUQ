"""The batched ChainUQ forward must reproduce the original per-sample forward."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.heads import CHAINUQ_FAMILY, ChainUQWrapper


def _batch(B=5, T=37, F=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(8, T + 1, (B,), generator=g)
    lens[0] = T
    attn = (torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)).long()
    cmask = torch.zeros(B, T)
    for i, n in enumerate(lens.tolist()):
        s = int(torch.randint(1, n - 2, (1,), generator=g))
        cmask[i, s: min(n, s + 4)] = 1.0
    tok = torch.randn(B, T, F, generator=g) * attn.unsqueeze(-1)
    return {"tok": tok, "attn": attn, "concl_mask": cmask}


@pytest.mark.parametrize("method", sorted(CHAINUQ_FAMILY))
def test_batched_matches_reference(method):
    torch.manual_seed(1)
    head = ChainUQWrapper(CHAINUQ_FAMILY[method], feature_dim=48, head_dim=32).eval()
    batch = _batch()
    with torch.no_grad():
        fast = head(batch)
        ref = head.reference_forward(batch)
    assert fast.shape == ref.shape
    torch.testing.assert_close(fast, ref, rtol=1e-4, atol=1e-5)
