"""Prompt construction: dataset system prompts plus the shared output contract."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

OUTPUT_CONTRACT = (
    "STRICT OUTPUT CONTRACT:\n"
    "Objective: solve the task using minimal evidence-based reasoning.\n"
    "Output schema (exact):\n"
    "Reasoning:\n"
    "Step 1: <one atomic fact>\n"
    "Step 2: <one atomic fact>\n"
    "...\n"
    "Conclusion: <dataset-specific final answer>\n"
    "Rules:\n"
    "- Plain text only. No markdown, bullets, or code fences.\n"
    "- 1 to 8 Step lines; use only as many steps as the evidence requires.\n"
    "- The Conclusion line must contain only the final answer format requested by the dataset.\n"
    "- No text before 'Reasoning:' or after 'Conclusion:'.\n"
    "Format example only:\n"
    "Reasoning:\n"
    "Step 1: <one evidence statement>\n"
    "Conclusion: <final answer>"
)


def build_system_prompt(*, objective: str, conclusion_schema: str, examples: Sequence[dict] | None = None,
                        max_steps: int = 6, max_step_words: int = 18, extra_rules: Iterable[str] | None = None) -> str:
    """Dataset-specific system prompt: objective, step schema, formatting rules, and few-shot examples."""
    lines: list[str] = [
        "You are a careful reasoning assistant.",
        f"Task objective: {objective}",
        "",
        "Output schema (must follow exactly):",
        "Reasoning:",
        "Step 1: <one atomic fact>",
        "Step 2: <one atomic fact>",
        "...",
        f"Conclusion: {conclusion_schema}",
        "",
        "Formatting constraints:",
        "- Plain text only; no markdown, code fences, bullets, or extra sections.",
        f"- Use 1-{max_steps} Step lines; use only as many steps as the evidence requires.",
        f"- Keep each Step line <= {max_step_words} words.",
        "- The Conclusion line must contain only the final answer format requested above.",
        "- Do not output any text before 'Reasoning:' or after the Conclusion line.",
    ]
    for rule in extra_rules or []:
        lines.append(f"- {str(rule).strip()}")
    if examples:
        lines += ["", "Few-shot examples:"]
        for idx, ex in enumerate(examples, start=1):
            sample_input, sample_output = str(ex.get("input", "")).strip(), str(ex.get("output", "")).strip()
            if sample_input and sample_output:
                lines += [f"Example {idx}", "Input:", sample_input, "Output:", sample_output]
    return "\n".join(lines)


def build_messages(ds, raw_item: dict) -> List[Dict[str, str]]:
    """Chat messages of one question: the dataset prompt with the output contract appended to the system turn."""
    messages = [dict(m) for m in ds.build_chat_messages(raw_item)]
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = f'{msg["content"].rstrip()}\n\n{OUTPUT_CONTRACT}'
            break
    else:
        messages.insert(0, {"role": "system", "content": OUTPUT_CONTRACT})
    return messages


def render_prompt_ids(tokenizer, messages, chat_template_kwargs: Optional[dict] = None) -> List[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        **(chat_template_kwargs or {}))
    if hasattr(ids, "keys") and "input_ids" in ids.keys():  # BatchEncoding in transformers 5
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]
