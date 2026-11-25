"""Shared UQ base rewritten for conclusion-only supervision."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.heads.base import UncertaintyHeadBase
from models.heads.claim_utils import prepare_claim_mask
from models.heads.uq_modules import CrossContextGate, ResidualMLP, resolve_heads


class UQHeadBase(UncertaintyHeadBase):
    supports_claim_inputs = True

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        *,
        head_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(feature_dim, num_classes)
        _ = kwargs
        _ = n_layers
        self.head_dim = int(head_dim)

        self.input_norm = nn.LayerNorm(feature_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sample_gate = nn.Linear(head_dim * 2, head_dim)
        self.sample_summary_proj = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sample_refine = ResidualMLP(head_dim=head_dim, dropout=dropout, expansion=2)
        self.logit_l2_penalty = 0.01

        attn_heads = resolve_heads(head_dim, n_heads)
        self.claim_type_embedding = nn.Embedding(3, head_dim)
        self.conclusion_marker_embedding = nn.Embedding(2, head_dim)
        self.mask_stats_proj = nn.Sequential(
            nn.Linear(4, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        scope_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=attn_heads,
            dim_feedforward=head_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.scope_encoder = nn.TransformerEncoder(scope_layer, num_layers=max(1, int(n_layers)))
        self.scope_norm = nn.LayerNorm(head_dim)
        self.conclusion_seed_proj = nn.Sequential(
            nn.Linear(head_dim * 4, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_gate = CrossContextGate(head_dim=head_dim, dropout=dropout)
        self.conclusion_refine = ResidualMLP(head_dim=head_dim, dropout=dropout, expansion=2)

        self.final_gate = nn.Linear(head_dim * 4, head_dim)
        self.final_refine = ResidualMLP(head_dim=head_dim, dropout=dropout, expansion=2)
        self.output_classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )
        self.logit_temperature = nn.Parameter(torch.tensor(0.7))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.claim_type_embedding.weight, std=0.02)
        nn.init.normal_(self.conclusion_marker_embedding.weight, std=0.02)

    def _project_tokens(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.token_proj(self.input_norm(token_features))

    def _masked_max_pool(
        self,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask_i is None:
            return token_h.max(dim=0).values
        mask = attention_mask_i.to(dtype=torch.bool).unsqueeze(-1)
        masked = token_h.masked_fill(~mask, torch.finfo(token_h.dtype).min)
        pooled = masked.max(dim=0).values
        mean_fallback = self.pool_features(token_h.unsqueeze(0), attention_mask_i.unsqueeze(0)).squeeze(0)
        return torch.where(torch.isfinite(pooled), pooled, mean_fallback)

    def _build_sample_state(
        self,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
    ) -> torch.Tensor:
        mean_pool = self.pool_features(
            token_h.unsqueeze(0),
            attention_mask_i.unsqueeze(0) if attention_mask_i is not None else None,
        ).squeeze(0)
        max_pool = self._masked_max_pool(token_h, attention_mask_i)
        gate = torch.sigmoid(self.sample_gate(torch.cat([mean_pool, max_pool], dim=-1)))
        fused = gate * mean_pool + (1.0 - gate) * max_pool
        sample_h = self.sample_summary_proj(torch.cat([fused, torch.abs(mean_pool - max_pool)], dim=-1))
        return self.sample_refine(sample_h)

    def _build_conclusion_state(
        self,
        *,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
        claim_mask_i: torch.Tensor,
        claim_types,
        sample_idx: int,
    ):
        device = token_h.device
        seq_len = token_h.shape[0]
        cm = prepare_claim_mask(
            claim_mask_i,
            seq_len=seq_len,
            device=device,
            attention_mask_i=attention_mask_i,
        )
        if cm.numel() == 0 or cm.shape[0] == 0:
            return None

        conclusion_idx = 0
        if claim_types is not None:
            type_row = claim_types[sample_idx].view(-1)
            limit = min(int(type_row.numel()), int(cm.shape[0]))
            for local_idx in range(limit):
                if int(type_row[local_idx].item()) == 1:
                    conclusion_idx = local_idx
                    break

        conclusion_mask = cm[conclusion_idx]
        if not bool(conclusion_mask.to(dtype=torch.bool).any().item()):
            return None
        denom = conclusion_mask.sum().clamp(min=1.0)
        conclusion_mean = (conclusion_mask.unsqueeze(-1) * token_h).sum(dim=0) / denom

        expanded = token_h.masked_fill(~conclusion_mask.to(dtype=torch.bool).unsqueeze(-1), torch.finfo(token_h.dtype).min)
        conclusion_max = expanded.max(dim=0).values
        conclusion_max = torch.where(torch.isfinite(conclusion_max), conclusion_max, conclusion_mean)

        claim_type_id = 1
        type_h = self.claim_type_embedding.weight[claim_type_id]

        stats_h = self._build_mask_stats(conclusion_mask, attention_mask_i)
        return conclusion_mean, conclusion_max, type_h, claim_type_id, conclusion_mask, stats_h

    def _build_mask_stats(
        self,
        conclusion_mask: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
    ) -> torch.Tensor:
        mask = conclusion_mask.to(dtype=torch.float32, device=conclusion_mask.device)
        valid = (
            attention_mask_i.to(dtype=torch.float32, device=conclusion_mask.device)
            if attention_mask_i is not None
            else torch.ones_like(mask)
        )
        valid_len = valid.sum().clamp(min=1.0)
        span_len = mask.sum().clamp(min=1.0)
        positions = torch.arange(mask.shape[0], device=mask.device, dtype=torch.float32)
        active = mask > 0
        if bool(active.any().item()):
            start = positions[active].min()
            end = positions[active].max()
            center = (positions * mask).sum() / span_len
        else:
            start = torch.zeros((), dtype=torch.float32, device=mask.device)
            end = torch.zeros((), dtype=torch.float32, device=mask.device)
            center = torch.zeros((), dtype=torch.float32, device=mask.device)
        denom = (valid_len - 1.0).clamp(min=1.0)
        stats = torch.stack(
            [
                span_len / valid_len,
                torch.log1p(span_len) / torch.log1p(valid_len),
                center / denom,
                (end - start + 1.0).clamp(min=1.0) / valid_len,
            ],
            dim=0,
        )
        return self.mask_stats_proj(stats.to(dtype=self.claim_type_embedding.weight.dtype))

    def _masked_mean_pool_1d(
        self,
        token_h: torch.Tensor,
        mask_1d: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask_1d.to(device=token_h.device, dtype=token_h.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (token_h * mask.unsqueeze(-1)).sum(dim=0) / denom

    def _masked_max_pool_1d(
        self,
        token_h: torch.Tensor,
        mask_1d: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask_1d.to(device=token_h.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return fallback
        masked = token_h.masked_fill(~mask.unsqueeze(-1), torch.finfo(token_h.dtype).min)
        pooled = masked.max(dim=0).values
        return torch.where(torch.isfinite(pooled), pooled, fallback)

    def _encode_conclusion_scope(
        self,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
        conclusion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        marker_ids = conclusion_mask.to(device=token_h.device, dtype=torch.long).clamp(min=0, max=1)
        marked = token_h + self.conclusion_marker_embedding(marker_ids)
        key_padding_mask = None
        if attention_mask_i is not None:
            key_padding_mask = ~attention_mask_i.to(device=token_h.device, dtype=torch.bool).unsqueeze(0)
        encoded = self.scope_encoder(marked.unsqueeze(0), src_key_padding_mask=key_padding_mask).squeeze(0)
        encoded = self.scope_norm(encoded)
        scope_global = self._build_sample_state(encoded, attention_mask_i)
        scope_mean = self._masked_mean_pool_1d(encoded, conclusion_mask)
        scope_max = self._masked_max_pool_1d(encoded, conclusion_mask, scope_mean)
        return scope_global, scope_mean, scope_max

    def _inject_token_context(
        self,
        conclusion_seed: torch.Tensor,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
    ) -> torch.Tensor:
        _ = token_h
        _ = attention_mask_i
        return conclusion_seed

    def _refine_conclusion_state(
        self,
        conclusion_h: torch.Tensor,
        sample_h: torch.Tensor,
        token_h: torch.Tensor,
        attention_mask_i: torch.Tensor | None,
        claim_type_id: int,
    ) -> torch.Tensor:
        _ = token_h
        _ = attention_mask_i
        _ = claim_type_id
        refined = self.context_gate(conclusion_h.unsqueeze(0), sample_h).squeeze(0)
        return self.conclusion_refine(refined)

    def _fuse_sample_and_conclusion(
        self,
        sample_h: torch.Tensor,
        conclusion_h: torch.Tensor,
    ) -> torch.Tensor:
        fusion_input = torch.cat(
            [sample_h, conclusion_h, torch.abs(sample_h - conclusion_h), sample_h * conclusion_h],
            dim=-1,
        )
        gate = torch.sigmoid(self.final_gate(fusion_input))
        fused = gate * conclusion_h + (1.0 - gate) * sample_h
        return self.final_refine(fused)

    def _classify_state(self, fused_h: torch.Tensor) -> torch.Tensor:
        logits = self.output_classifier(fused_h)
        temperature = 1.0 + torch.nn.functional.softplus(self.logit_temperature)
        return logits / temperature

    def _forward_without_claims(self, features: torch.Tensor, attention_mask: torch.Tensor | None):
        outputs = []
        for sample_idx in range(features.shape[0]):
            attn_i = attention_mask[sample_idx] if attention_mask is not None else None
            token_h = self._project_tokens(features[sample_idx])
            sample_h = self._build_sample_state(token_h, attn_i)
            outputs.append(self._classify_state(sample_h))
        return torch.stack(outputs, dim=0)

    def forward(self, features, attention_mask=None, claim_masks=None, claim_types=None):
        if claim_masks is None:
            return self._forward_without_claims(features, attention_mask)

        outputs = []
        for sample_idx in range(features.shape[0]):
            attn_i = attention_mask[sample_idx] if attention_mask is not None else None
            token_h = self._project_tokens(features[sample_idx])
            sample_h = self._build_sample_state(token_h, attn_i)

            encoded = self._build_conclusion_state(
                token_h=token_h,
                attention_mask_i=attn_i,
                claim_mask_i=claim_masks[sample_idx],
                claim_types=claim_types,
                sample_idx=sample_idx,
            )
            if encoded is None:
                outputs.append(self._classify_state(sample_h))
                continue

            _conclusion_mean, _conclusion_max, type_h, claim_type_id, conclusion_mask, stats_h = encoded
            scope_global, scope_mean, scope_max = self._encode_conclusion_scope(
                token_h=token_h,
                attention_mask_i=attn_i,
                conclusion_mask=conclusion_mask,
            )
            conclusion_seed = self.conclusion_seed_proj(
                torch.cat(
                    [
                        scope_mean,
                        scope_max,
                        scope_global,
                        stats_h + type_h + torch.abs(scope_mean - scope_global),
                    ],
                    dim=-1,
                )
            )
            conclusion_h = self._inject_token_context(conclusion_seed, token_h, attn_i)
            conclusion_h = self._refine_conclusion_state(
                conclusion_h=conclusion_h,
                sample_h=scope_global,
                token_h=token_h,
                attention_mask_i=attn_i,
                claim_type_id=claim_type_id,
            )
            fused_h = self._fuse_sample_and_conclusion(scope_global, conclusion_h)
            outputs.append(self._classify_state(fused_h))

        return torch.stack(outputs, dim=0)
