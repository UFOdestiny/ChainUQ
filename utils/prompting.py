"""Helpers for safe chat-template prompt construction."""

from collections.abc import Sequence
from typing import Dict, List, Union

import torch

PromptInput = Union[str, List[int]]


def normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Coerce chat messages into the minimal role/content format."""
    safe_messages = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        safe_messages.append(
            {
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", "")),
            }
        )
    return safe_messages


def messages_to_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    """Serialize chat messages into a plain-text fallback prompt."""
    parts = []
    for msg in normalize_chat_messages(messages):
        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def build_chat_prompt_input(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> PromptInput:
    """Build a generation prompt as token ids when chat templates support it."""
    safe_messages = normalize_chat_messages(messages)
    try:
        prompt_ids = tokenizer.apply_chat_template(
            safe_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids.get("input_ids", prompt_ids)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        if (
            isinstance(prompt_ids, list)
            and prompt_ids
            and isinstance(prompt_ids[0], Sequence)
            and not isinstance(prompt_ids[0], (int, float))
        ):
            prompt_ids = prompt_ids[0]
        if isinstance(prompt_ids, list):
            return [int(token_id) for token_id in prompt_ids]
    except Exception:
        pass
    return messages_to_fallback_prompt(safe_messages)


def prompt_to_token_ids(tokenizer, prompt: PromptInput) -> List[int]:
    """Normalize a prompt input into a flat list of token ids."""
    if isinstance(prompt, list):
        return [int(token_id) for token_id in prompt]
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if (
        isinstance(token_ids, list)
        and token_ids
        and isinstance(token_ids[0], Sequence)
        and not isinstance(token_ids[0], (int, float))
    ):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def prompt_token_length(tokenizer, prompt: PromptInput) -> int:
    """Return the token length of a prompt input."""
    return len(prompt_to_token_ids(tokenizer, prompt))


def truncate_token_ids(tokenizer, token_ids: List[int], max_length: int) -> List[int]:
    """Truncate token ids using the tokenizer's truncation side."""
    ids = [int(token_id) for token_id in token_ids]
    if max_length <= 0 or len(ids) <= max_length:
        return ids
    if getattr(tokenizer, "truncation_side", "right") == "left":
        return ids[-max_length:]
    return ids[:max_length]


def build_padded_token_batch(
    rows: List[List[int]],
    *,
    pad_token_id: int,
    device,
):
    """Pad token-id rows into ``input_ids`` and ``attention_mask`` tensors."""
    max_len = max((len(row) for row in rows), default=0)
    max_len = max(max_len, 1)
    input_ids = torch.full(
        (len(rows), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(rows), max_len),
        dtype=torch.long,
        device=device,
    )
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[row_idx, : len(row)] = row_tensor
        attention_mask[row_idx, : len(row)] = 1
    return input_ids, attention_mask
