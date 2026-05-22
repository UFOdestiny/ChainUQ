"""Hybrid vLLM + HuggingFace generation engine.

vLLM handles fast text generation while an optional HF model provides
hidden-state / attention features needed for uncertainty quantification.

IMPORTANT: the VLLM_WORKER_MULTIPROC_METHOD env-var and the spawn
start-method **must** be set before any vLLM import.
"""
import multiprocessing
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from typing import Any, Dict, List, Optional, Set

import torch

from config import Config
from engine.types import GenerationResult
from models.features.combined import CombinedExtractor
from utils.common import load_tokenizer_from_path, resolve_torch_dtype
from utils.efficiency import load_model_with_dtype
from utils.log import get_logger
from utils.prompting import (
    PromptInput,
    build_padded_token_batch,
    prompt_to_token_ids,
    truncate_token_ids,
)

log = get_logger(__name__)

class _SelectiveAttentionManager:
    """Hook-based manager that enables eager attention only for selected layers.

    When ``output_attentions=True`` is passed to the full model, HF forces
    **every** decoder layer into eager (non-FlashAttention) mode and stores
    the full (batch, heads, seq, seq) attention weight matrix.  For 32-layer
    models this is catastrophic for GPU memory.

    This manager installs forward pre-hooks and post-hooks on the *non-selected*
    layers so that:

    * **Pre-hook** overrides ``output_attentions`` to ``False`` → the layer
      uses FlashAttention and does NOT materialise the attention matrix.
    * **Post-hook** inserts a ``None`` placeholder at position 1 in the layer
      output tuple so that the outer model's bookkeeping (which expects index-1
      to be attention weights when ``output_attentions=True`` at model level)
      works without IndexError.

    Selected layers are left untouched so they run with eager attention and
    return real attention weight tensors.
    """

    def __init__(self, hf_model: torch.nn.Module, keep_layer_indices: List[int]):
        self._handles: list = []

        # Resolve the list of decoder layers (works for Mistral, Llama, Qwen, Gemma)
        layers = _get_decoder_layers(hf_model)
        num_layers = len(layers)
        keep_set: Set[int] = {
            idx % num_layers for idx in keep_layer_indices
        }

        for i, layer in enumerate(layers):
            if i not in keep_set:
                h1 = layer.register_forward_pre_hook(
                    self._disable_attn, with_kwargs=True,
                )
                h2 = layer.register_forward_hook(
                    self._inject_none, with_kwargs=True,
                )
                self._handles.extend([h1, h2])

    # --- hooks -----------------------------------------------------------

    @staticmethod
    def _disable_attn(_module, args, kwargs):
        kwargs["output_attentions"] = False
        return args, kwargs

    @staticmethod
    def _inject_none(_module, _args, _kwargs, output):
        # Layer with output_attentions=False returns (hidden_states, ...) without
        # an attention slot.  Insert None at position 1 so the model-level loop
        # that does ``all_self_attns += (layer_outputs[1],)`` finds None there.
        if isinstance(output, tuple):
            return (output[0], None) + output[1:]
        return output

    # --- lifecycle -------------------------------------------------------

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def _get_decoder_layers(hf_model):
    """Return the nn.ModuleList of decoder layers from a CausalLM model."""
    # Most HF CausalLM wrappers: model.model.layers
    inner = getattr(hf_model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(hf_model).__name__}. "
        "Expected model.model.layers (Mistral/Llama/Qwen/Gemma pattern)."
    )


class VLLMGenerationEngine:
    """vLLM for generation, optional HF model for feature extraction."""

    def __init__(self, cfg: Config, model_path: str, device: torch.device,
                 text_only: bool = False):
        self.cfg = cfg
        self.device = device
        self.model_path = model_path
        self.text_only = bool(text_only)

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

        # --- vLLM engine (loaded FIRST to pre-allocate GPU memory) ---
        attn_backend = cfg.vllm.attention_backend
        log.info(
            "Loading vLLM engine from %s (attention_backend=%s)...",
            model_path, attn_backend,
        )
        from vllm import LLM, SamplingParams

        gpu_util = cfg.vllm.gpu_memory_utilization
        if not self.text_only and gpu_util > 0.4:
            # When both vLLM and HF model share the GPU, limit vLLM's KV cache
            # to leave room for HF model forward passes (attention, activations).
            gpu_util = 0.4
        log.info("gpu_memory_utilization=%.2f (text_only=%s)", gpu_util, self.text_only)

        self._vllm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=cfg.vllm.max_model_len,
            enforce_eager=cfg.vllm.enforce_eager,
            swap_space=cfg.vllm.swap_space,
            dtype=cfg.vllm.dtype,
            seed=cfg.vllm.seed,
            trust_remote_code=cfg.model.trust_remote_code,
            disable_log_stats=True,
            block_size=32,
            attention_backend=attn_backend,
        )
        self._SamplingParams = SamplingParams
        log.info("vLLM engine loaded successfully.")

        # --- HF model for feature extraction (optional) ---
        self._hf_model = None
        if not self.text_only:
            log.info(
                "Loading HF model for feature extraction from %s ...",
                model_path,
            )
            torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)
            self._hf_model = load_model_with_dtype(
                AutoModelForCausalLM.from_pretrained,
                model_path,
                torch_dtype,
                device_map=cfg.model.device_map,
                trust_remote_code=cfg.model.trust_remote_code,
                cache_dir=cfg.model.hf_cache_dir,
                attn_implementation="eager",
            )
            self._hf_model.eval()
        else:
            log.info("Text-only mode: skipping HF model load.")

    def unload(self):
        """Explicitly release GPU memory held by vLLM and HF models.

        Call this when the engine is no longer needed so that subsequent
        stages can use the freed VRAM.
        """
        import gc

        if self._hf_model is not None:
            self._hf_model.cpu()
            del self._hf_model
            self._hf_model = None

        if self._vllm_engine is not None:
            # Explicitly shut down the vLLM engine core subprocess to free GPU
            try:
                engine_core = getattr(self._vllm_engine.llm_engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    log.info("Shutting down vLLM engine core (%s)...", type(engine_core).__name__)
                    engine_core.shutdown()
                    log.info("vLLM engine core shutdown complete.")
                else:
                    log.warning("No engine_core.shutdown() available (type=%s).",
                                type(engine_core).__name__ if engine_core else "None")
            except Exception as e:
                log.warning("engine_core.shutdown() raised %s: %s", type(e).__name__, e)
            del self._vllm_engine
            self._vllm_engine = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Engine unloaded — GPU memory released.")

    def __del__(self):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    @property
    def model_config(self):
        if self._hf_model is None:
            raise RuntimeError("model_config unavailable in text_only mode.")
        return self._hf_model.config

    # ------------------------------------------------------------------
    # Generation with feature extraction
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[PromptInput],
        feature_extractor: CombinedExtractor,
        max_new_tokens: int,
        use_attention: bool,
        attn_layer_indices: Optional[List[int]] = None,
    ) -> GenerationResult:
        """Generate via vLLM, then run HF forward pass for features."""
        # Safety: respect save_attention_weights config as master switch
        if not self.cfg.generation.save_attention_weights:
            use_attention = False

        if self.text_only or self._hf_model is None:
            raise RuntimeError(
                "generate_batch() requires HF model; init with text_only=False."
            )

        # Pre-filter prompts that exceed vLLM context window
        max_prompt_tokens = self.cfg.vllm.max_model_len - max_new_tokens
        prompt_token_rows = [
            prompt_to_token_ids(self.tokenizer, prompt)
            for prompt in prompts
        ]
        prompt_token_lengths = [len(ids) for ids in prompt_token_rows]

        valid_indices = []
        valid_prompts = []
        valid_prompt_token_rows = []
        for i, tok_len in enumerate(prompt_token_lengths):
            if tok_len <= max_prompt_tokens:
                valid_indices.append(i)
                valid_prompts.append({"prompt_token_ids": prompt_token_rows[i]})
                valid_prompt_token_rows.append(prompt_token_rows[i])
            else:
                log.warning(
                    "Skipping sample %d: prompt has %d tokens, max allowed is %d "
                    "(max_model_len=%d - max_new_tokens=%d)",
                    i, tok_len, max_prompt_tokens,
                    self.cfg.vllm.max_model_len, max_new_tokens,
                )

        sampling_params = self._SamplingParams(
            max_tokens=max_new_tokens, temperature=0, n=1,
        )

        # Generate only for valid prompts
        if valid_prompts:
            vllm_outputs = self._vllm_engine.generate(
                valid_prompts, sampling_params, use_tqdm=False,
            )
        else:
            vllm_outputs = []

        # Reconstruct full batch (None for skipped samples)
        generated_texts = [""] * len(prompts)
        full_token_rows = [[] for _ in prompts]
        gen_tok_list = [[] for _ in prompts]
        prompt_lens = [
            len(
                truncate_token_ids(
                    self.tokenizer,
                    token_ids,
                    self.cfg.model.model_max_length,
                )
            )
            for token_ids in prompt_token_rows
        ]
        for output, orig_idx, prompt_ids in zip(
            vllm_outputs,
            valid_indices,
            valid_prompt_token_rows,
        ):
            gen_output = output.outputs[0]
            gen_text = gen_output.text
            gen_token_ids = [int(token_id) for token_id in getattr(gen_output, "token_ids", [])]
            generated_texts[orig_idx] = gen_text
            gen_tok_list[orig_idx] = gen_token_ids
            full_token_rows[orig_idx] = truncate_token_ids(
                self.tokenizer,
                prompt_ids + gen_token_ids,
                self.cfg.model.model_max_length,
            )

        skipped_set = set(range(len(prompts))) - set(valid_indices)

        # --- HF forward in sub-batches to avoid OOM ---
        hf_micro = getattr(self.cfg.generation, "hf_micro_batch_size", 0) or 2
        batch_features, batch_top_probs, batch_log_likelihoods = (
            [None] * len(prompts),
            [None] * len(prompts),
            [None] * len(prompts),
        )

        # Indices that actually need HF forward (not skipped, not empty)
        need_hf = [i for i in range(len(prompts)) if i not in skipped_set and full_token_rows[i]]

        # If using attention, install selective hooks so only requested layers
        # use eager attention (others keep FlashAttention).
        attn_mgr = None
        if use_attention and attn_layer_indices is not None:
            attn_mgr = _SelectiveAttentionManager(self._hf_model, attn_layer_indices)
            log.debug("Selective attention enabled for layers %s", attn_layer_indices)

        try:
            pos = 0
            cur_micro = hf_micro
            while pos < len(need_hf):
                mb_indices = need_hf[pos : pos + cur_micro]
                mb_full_rows = [full_token_rows[i] for i in mb_indices]
                mb_input_ids, mb_attention_mask = build_padded_token_batch(
                    mb_full_rows,
                    pad_token_id=self.tokenizer.pad_token_id,
                    device=self.device,
                )

                try:
                    with torch.no_grad():
                        hf_out = self._hf_model(
                            input_ids=mb_input_ids,
                            attention_mask=mb_attention_mask,
                            output_hidden_states=True,
                            output_attentions=use_attention,
                            use_cache=False,
                            return_dict=True,
                        )
                except torch.cuda.OutOfMemoryError:
                    del mb_input_ids, mb_attention_mask
                    torch.cuda.empty_cache()
                    if cur_micro <= 1:
                        log.warning(
                            "HF forward OOM at micro-batch=1 — cannot reduce further; re-raise",
                        )
                        raise  # already at 1, cannot reduce further
                    prev_micro = cur_micro
                    cur_micro = max(1, cur_micro // 2)
                    log.info(
                        "HF forward OOM — reducing HF micro-batch %d → %d and retrying",
                        prev_micro,
                        cur_micro,
                    )
                    continue  # retry same position with smaller batch

                for j, orig_idx in enumerate(mb_indices):
                    p_len = min(prompt_lens[orig_idx], int(mb_attention_mask[j].sum().item()))
                    total_len = int(mb_attention_mask[j].sum().item())
                    gen_len = total_len - p_len
                    if gen_len <= 0:
                        continue

                    gen_hidden = tuple(
                        layer_hs[j : j + 1, p_len:total_len, :]
                        for layer_hs in hf_out.hidden_states
                    )
                    gen_logits = hf_out.logits[j : j + 1, p_len:total_len, :]

                    gen_attentions = None
                    if use_attention and hf_out.attentions is not None:
                        gen_attentions = tuple(
                            la[j : j + 1, :, p_len:total_len, :total_len]
                            if la is not None
                            else None
                            for la in hf_out.attentions
                        )

                    gen_mask = torch.ones(1, gen_len, dtype=torch.long, device=self.device)
                    features = feature_extractor(
                        hidden_states=gen_hidden,
                        logits=gen_logits,
                        attentions=gen_attentions,
                        attention_mask=gen_mask,
                    ).to(torch.bfloat16)
                    batch_features[orig_idx] = features

                    log_probs = torch.log_softmax(gen_logits.float(), dim=-1)
                    gen_tids = mb_input_ids[j, p_len:total_len]
                    ll = (
                        log_probs[0]
                        .gather(dim=-1, index=gen_tids.unsqueeze(-1))
                        .squeeze(-1)
                        .to(torch.bfloat16)
                    )
                    tp = (
                        torch.softmax(gen_logits[0].float(), dim=-1)
                        .topk(4, dim=-1)
                        .values.to(torch.bfloat16)
                    )
                    batch_top_probs[orig_idx] = tp
                    batch_log_likelihoods[orig_idx] = ll

                del hf_out, mb_input_ids, mb_attention_mask
                torch.cuda.empty_cache()
                pos += cur_micro
        finally:
            if attn_mgr is not None:
                attn_mgr.remove()

        features_out, top_probs_out, log_ll_out = _pad_batch_results(
            batch_features, batch_top_probs, batch_log_likelihoods, self.device,
        )

        max_gen_len = 0
        for ids in gen_tok_list:
            max_gen_len = max(max_gen_len, len(ids))
        if max_gen_len == 0:
            max_gen_len = 1
        generated_token_ids = torch.zeros(
            len(prompts), max_gen_len, dtype=torch.long, device=self.device,
        )
        for i, ids in enumerate(gen_tok_list):
            if ids:
                generated_token_ids[i, :len(ids)] = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.device,
                )
        generated_lengths = [int(len(ids)) for ids in gen_tok_list]

        return GenerationResult(
            generated_texts=generated_texts,
            generated_lengths=generated_lengths,
            generated_token_ids=generated_token_ids,
            features=features_out,
            top_probs=top_probs_out,
            log_likelihoods=log_ll_out,
        )

    # ------------------------------------------------------------------
    # Text-only generation (vLLM only, no HF forward pass)
    # ------------------------------------------------------------------

    def generate_text_only(
        self,
        prompts: List[PromptInput],
        max_new_tokens: int,
        structured_json_schema: Optional[Dict[str, Any]] = None,
        thinking_token_budget: Optional[int] = None,
    ) -> List[str]:
        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "n": 1,
        }
        if thinking_token_budget is not None:
            sampling_kwargs["thinking_token_budget"] = int(thinking_token_budget)
        if structured_json_schema is not None:
            try:
                from vllm.sampling_params import StructuredOutputsParams
            except Exception as exc:
                raise RuntimeError(
                    "Current vLLM build does not expose StructuredOutputsParams; "
                    "upgrade vLLM to a version that supports structured outputs."
                ) from exc
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=structured_json_schema
            )
        sampling_params = self._SamplingParams(**sampling_kwargs)
        vllm_prompts = [
            {"prompt_token_ids": prompt_to_token_ids(self.tokenizer, prompt)}
            for prompt in prompts
        ]
        vllm_outputs = self._vllm_engine.generate(
            vllm_prompts, sampling_params, use_tqdm=False,
        )
        return [output.outputs[0].text for output in vllm_outputs]

def _pad_batch_results(batch_features, batch_top_probs, batch_log_likelihoods, device):
    """Zero-pad per-sample features / probs / log-likelihoods to uniform length."""
    valid = [i for i, f in enumerate(batch_features) if f is not None]
    if not valid:
        return None, None, None

    max_gen_len = max(batch_features[i].shape[1] for i in valid)
    feat_dim = batch_features[valid[0]].shape[-1]
    batch_size = len(batch_features)

    features_out = torch.zeros(batch_size, max_gen_len, feat_dim, device=device)
    top_probs_out = torch.zeros(batch_size, max_gen_len, 4, device=device)
    log_ll_out = torch.zeros(batch_size, max_gen_len, device=device)

    for i in valid:
        g = batch_features[i].shape[1]
        features_out[i, :g, :] = batch_features[i][0]
        if batch_top_probs[i] is not None:
            top_probs_out[i, :g, :] = batch_top_probs[i][:g]
        if batch_log_likelihoods[i] is not None:
            log_ll_out[i, :g] = batch_log_likelihoods[i][:g]

    return features_out, top_probs_out, log_ll_out
