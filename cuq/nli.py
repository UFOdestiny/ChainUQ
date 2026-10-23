"""DeBERTa-large-MNLI helper (CCP, Semantic Entropy, SelfCheck-NLI, NLI step verifier)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from cuq.registry import NLI_MODEL


class NLI:
    def __init__(self, device="cuda", batch_size: int = 256, max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL.path).to(device).eval().half()
        self.device, self.bs, self.max_length = device, batch_size, max_length
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.idx = (labels["contradiction"], labels["neutral"], labels["entailment"])

    @torch.inference_mode()
    def probs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Return [N, 3] probabilities ordered (contradiction, neutral, entailment)."""
        out = []
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for k in range(0, len(order), self.bs):
            idx = order[k: k + self.bs]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits.float(), -1)[:, list(self.idx)].cpu().numpy()
            out.extend(zip(idx, p))
        res = np.zeros((len(pairs), 3), dtype=np.float32)
        for i, p in out:
            res[i] = p
        return res
