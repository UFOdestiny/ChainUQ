"""Model wrappers for cached-feature supervision and evaluation."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

@dataclass
class ClaimModelOutput(ModelOutput):
    """Trainer-compatible output for conclusion-only supervised heads."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    sample_logits: Optional[torch.Tensor] = None
    sample_loss: Optional[torch.Tensor] = None


class CachedFeatureModel(nn.Module):
    """Phase 2 model: trains head on pre-extracted features (no LLM needed).

    Takes pre-extracted feature tensors directly, skipping the LLM forward pass.
    Uses BCEWithLogitsLoss for binary classification (num_classes=1).
    """

    def __init__(
        self,
        head: nn.Module,
        num_classes: int = 1,
        loss_type: str = "bce",
        pos_weight: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        sample_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.head = head
        self.num_classes = num_classes
        self.loss_type = (loss_type or "bce").lower()
        self.pos_weight = float(pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.sample_pos_weight = float(sample_pos_weight)

    def _binary_loss(
        self,
        logits_1d: torch.Tensor,
        labels_1d: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
        pos_weight_by_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels_1d = labels_1d.float()
        # Label smoothing: soften hard 0/1 targets to reduce overconfidence
        if self.label_smoothing > 0:
            labels_1d = labels_1d * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.loss_type in ("balanced_bce", "focal"):
            if pos_weight_by_sample is None:
                pos_w = torch.full_like(
                    labels_1d,
                    fill_value=max(self.pos_weight, 1e-8),
                    dtype=torch.float32,
                    device=logits_1d.device,
                )
            else:
                pos_w = pos_weight_by_sample.to(logits_1d.device).float().clamp(min=1e-8)

            pos_term = -labels_1d * nn.functional.logsigmoid(logits_1d) * pos_w
            neg_term = -(1.0 - labels_1d) * nn.functional.logsigmoid(-logits_1d)
            bce = pos_term + neg_term

        if self.loss_type == "focal":
            # Focal loss with class-balance weighting
            probs = torch.sigmoid(logits_1d)
            pt = torch.where(labels_1d > 0.5, probs, 1.0 - probs)
            focal_weight = (1.0 - pt).pow(self.focal_gamma)
            alpha = torch.where(labels_1d > 0.5, pos_w, torch.ones_like(pos_w))
            focal_weight = focal_weight * alpha
            if sample_weights is not None:
                return (focal_weight * sample_weights * bce).mean()
            return (focal_weight * bce).mean()

        if self.loss_type == "balanced_bce":
            if sample_weights is not None:
                return (bce * sample_weights).mean()
            return bce.mean()

        # Plain BCE
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits_1d, labels_1d, reduction="none",
        )
        if sample_weights is not None:
            return (bce * sample_weights).mean()
        return bce.mean()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        claim_masks=None,
        claim_types=None,
        claim_labels=None,
        **kwargs,
    ) -> ClaimModelOutput:
        if claim_masks is not None and claim_labels is not None:
            return self._forward_claim_level(
                features,
                attention_mask,
                claim_masks,
                claim_types,
                claim_labels,
                labels=labels,
            )

        logits = self.head(features, attention_mask)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss = self._binary_loss(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return ClaimModelOutput(loss=loss, logits=logits)

    def _extract_conclusion_logits(
        self,
        logits: torch.Tensor,
        claim_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract one supervised conclusion logit per sample.

        The training/eval dataset now carries exactly one usable conclusion claim
        per sample, so the supervised head output is used directly as the sample
        logit with no additional aggregation layer.
        """
        batch_size = int(claim_labels.shape[0])
        sample_parts: list[torch.Tensor] = []

        if logits.ndim == 3:
            for sample_idx in range(batch_size):
                n_claims = int(claim_labels[sample_idx].numel())
                current = logits[sample_idx, : max(1, n_claims), :].reshape(-1, logits.shape[-1])
                sample_parts.append(current[0])
        elif logits.ndim == 2:
            if logits.shape[0] == batch_size:
                sample_parts = [logits[sample_idx] for sample_idx in range(batch_size)]
            else:
                cursor = 0
                for sample_idx in range(batch_size):
                    n_claims = int(claim_labels[sample_idx].numel())
                    current = logits[cursor: cursor + max(1, n_claims), :]
                    sample_parts.append(current[0])
                    cursor += max(1, n_claims)
        else:
            flat_logits = logits.view(-1, logits.shape[-1])
            cursor = 0
            for sample_idx in range(batch_size):
                n_claims = int(claim_labels[sample_idx].numel())
                current = flat_logits[cursor: cursor + max(1, n_claims), :]
                sample_parts.append(current[0])
                cursor += max(1, n_claims)

        if not sample_parts:
            empty = logits.new_zeros((0, self.num_classes))
            return empty, empty

        sample_logits = torch.stack(sample_parts, dim=0)
        return sample_logits, sample_logits

    def _forward_claim_level(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_masks,
        claim_types,
        claim_labels,
        labels: Optional[torch.Tensor] = None,
    ) -> ClaimModelOutput:
        if getattr(self.head, "supports_claim_inputs", False):
            if hasattr(self.head, "forward_claim"):
                logits = self.head.forward_claim(
                    features=features,
                    attention_mask=attention_mask,
                    claim_masks=claim_masks,
                    claim_types=claim_types,
                )
            else:
                logits = self.head(
                    features=features,
                    attention_mask=attention_mask,
                    claim_masks=claim_masks,
                    claim_types=claim_types,
                )

            logits_flat, sample_logits = self._extract_conclusion_logits(logits, claim_labels)

            total_loss = logits_flat.sum() * 0.0
            sample_loss = None
            if labels is not None and sample_logits.numel() > 0:
                sample_labels = labels.to(features.device).float().view(-1)
                sample_pos_w = torch.full_like(
                    sample_labels,
                    fill_value=max(self.sample_pos_weight, 1e-8),
                    dtype=torch.float32,
                    device=sample_labels.device,
                )
                sample_loss = self._binary_loss(
                    logits_1d=sample_logits.squeeze(-1),
                    labels_1d=sample_labels,
                    sample_weights=None,
                    pos_weight_by_sample=sample_pos_w,
                )
                logit_l2_penalty = float(getattr(self.head, "logit_l2_penalty", 0.0) or 0.0)
                if logit_l2_penalty > 0:
                    sample_loss = sample_loss + logit_l2_penalty * sample_logits.squeeze(-1).float().pow(2).mean()
                total_loss = sample_loss

            return ClaimModelOutput(
                loss=total_loss,
                logits=logits_flat,
                sample_logits=sample_logits,
                sample_loss=sample_loss,
            )

        raise RuntimeError(
            f"{self.head.__class__.__name__} must consume claim inputs directly; "
            "wrapper-side pooled-claim fallback has been removed."
        )

    def get_trainable_params(self):
        return [p for p in self.head.parameters() if p.requires_grad]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        _ = exclude_embeddings
        return sum(p.numel() for p in self.get_trainable_params())
