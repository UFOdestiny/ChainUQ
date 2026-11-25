"""HuggingFace generation engine for Phase 1."""
from typing import List

import torch

from config import Config
from engine.types import GenerationResult
from models.features.combined import CombinedExtractor
from utils.common import load_llm_from_path, load_tokenizer_from_path
from utils.log import get_logger
from utils.prompting import (
    PromptInput,
    build_padded_token_batch,
    prompt_to_token_ids,
    truncate_token_ids,
)

log = get_logger(__name__)


def _generated_lengths_from_ids(tokenizer, gen_token_ids: torch.Tensor) -> list[int]:
    """Infer per-sample generated lengths by trimming trailing pad/eos tokens."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    lengths: list[int] = []
    for row in gen_token_ids:
        ids = row.tolist()
        end = len(ids)
        while end > 0 and pad_id is not None and ids[end - 1] == int(pad_id):
            end -= 1
        while end > 0 and eos_id is not None and ids[end - 1] == int(eos_id):
            end -= 1
        lengths.append(max(end, 0))
    return lengths


class HFGenerationEngine:
    """Wraps a HuggingFace causal-LM for generation + feature extraction."""

    def __init__(self, cfg: Config, model_path: str, device: torch.device):
        self.cfg = cfg
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading tokenizer from %s", model_path)
        self.tokenizer = load_tokenizer_from_path(
            tokenizer_cls=AutoTokenizer,
            model_path=model_path,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.hf_cache_dir,
            padding_side=cfg.model.padding_side,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info("Loading HF model from %s", model_path)
        self.llm = load_llm_from_path(
            model_cls=AutoModelForCausalLM,
            model_path=model_path,
            torch_dtype_name=cfg.model.torch_dtype,
            device_map=cfg.model.device_map,
            trust_remote_code=cfg.model.trust_remote_code,
            cache_dir=cfg.model.hf_cache_dir,
            attn_implementation="eager",
        )
        self.llm.eval()

    @property
    def model_config(self):
        return self.llm.config

    def generate_batch(
        self,
        prompts: List[PromptInput],
        feature_extractor: CombinedExtractor,
        max_new_tokens: int,
        use_attention: bool,
    ) -> GenerationResult:
        """Generate tokens and extract per-token features in one pass."""
        cfg = self.cfg
        prompt_rows = [
            truncate_token_ids(
                self.tokenizer,
                prompt_to_token_ids(self.tokenizer, prompt),
                cfg.model.model_max_length,
            )
            for prompt in prompts
        ]
        input_ids, attention_mask = build_padded_token_batch(
            prompt_rows,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature if cfg.generation.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=cfg.generation.save_hidden_states,
                output_attentions=cfg.generation.save_attention_weights and use_attention,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_token_ids = gen_outputs.sequences[:, prompt_len:]
        generated_texts = self.tokenizer.batch_decode(
            gen_token_ids, skip_special_tokens=True
        )
        generated_lengths = _generated_lengths_from_ids(self.tokenizer, gen_token_ids)

        features, top_probs, log_likelihoods = reconstruct_features_from_generation(
            gen_outputs, prompt_len, feature_extractor, inputs["attention_mask"], self.device
        )

        return GenerationResult(
            generated_texts=generated_texts,
            generated_lengths=generated_lengths,
            generated_token_ids=gen_token_ids,
            features=features,
            top_probs=top_probs,
            log_likelihoods=log_likelihoods,
        )

    def generate_text_only(
        self, prompts: List[PromptInput], max_new_tokens: int
    ) -> List[str]:
        """Generate text without feature extraction (greedy decoding)."""
        cfg = self.cfg
        prompt_rows = [
            truncate_token_ids(
                self.tokenizer,
                prompt_to_token_ids(self.tokenizer, prompt),
                cfg.model.model_max_length,
            )
            for prompt in prompts
        ]
        input_ids, attention_mask = build_padded_token_batch(
            prompt_rows,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_token_ids = gen_outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(gen_token_ids, skip_special_tokens=True)

def reconstruct_features_from_generation(
    gen_outputs, prompt_len, feature_extractor, attention_mask, device
):
    """Reassemble per-token hidden states, attentions, and logits into features.

    Returns ``(features, top_probs, log_likelihoods)`` or ``(None, None, None)``
    when no tokens were generated.
    """
    n_gen_tokens = len(gen_outputs.scores)
    if n_gen_tokens == 0:
        return None, None, None

    logits = torch.stack(gen_outputs.scores, dim=1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gen_token_ids = gen_outputs.sequences[:, prompt_len:]

    log_likelihoods = (
        log_probs.gather(dim=-1, index=gen_token_ids.unsqueeze(-1))
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    top_probs = (
        torch.softmax(logits.float(), dim=-1)
        .topk(4, dim=-1)
        .values.to(torch.bfloat16)
    )

    # --- hidden states: reassemble from per-step tuples ---
    n_layers = len(gen_outputs.hidden_states[0])
    hidden_states_per_layer = []
    for layer_idx in range(n_layers):
        step0 = gen_outputs.hidden_states[0][layer_idx][:, -1:, :]
        rest = [
            gen_outputs.hidden_states[t][layer_idx]
            for t in range(1, n_gen_tokens)
        ]
        layer_hs = torch.cat([step0] + rest, dim=1)
        hidden_states_per_layer.append(layer_hs)
    hidden_states_tuple = tuple(hidden_states_per_layer)

    # --- attentions (optional) ---
    attentions_tuple = None
    if gen_outputs.attentions is not None and len(gen_outputs.attentions) > 0:
        n_attn_layers = len(gen_outputs.attentions[0])
        batch_size = gen_outputs.sequences.shape[0]
        gen_seq_len = n_gen_tokens
        max_ctx = prompt_len + gen_seq_len
        n_heads = gen_outputs.attentions[0][0].shape[1]

        layer_attns = []
        for layer_idx in range(n_attn_layers):
            sample_attn = gen_outputs.attentions[0][layer_idx]
            attn_full = torch.zeros(
                batch_size, n_heads, gen_seq_len, max_ctx,
                device=device, dtype=sample_attn.dtype,
            )
            for t in range(gen_seq_len):
                a = gen_outputs.attentions[t][layer_idx]
                ctx_len_t = a.shape[-1]
                if a.dim() == 4:
                    step_attn = a[:, :, -1, :ctx_len_t]
                elif a.dim() == 3:
                    step_attn = a[:, :, :ctx_len_t]
                else:
                    raise RuntimeError(
                        f"Unexpected attention tensor rank={a.dim()}"
                    )
                attn_full[:, :, t, :ctx_len_t] = step_attn
            layer_attns.append(attn_full)
        attentions_tuple = tuple(layer_attns)

    gen_attn_mask = torch.ones(
        gen_token_ids.shape, dtype=torch.long, device=device
    )
    features = feature_extractor(
        hidden_states=hidden_states_tuple,
        logits=logits,
        attentions=attentions_tuple,
        attention_mask=gen_attn_mask,
    ).to(torch.bfloat16)

    return features, top_probs, log_likelihoods
