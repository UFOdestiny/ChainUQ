"""Prompt construction, reasoning-format variants, and claim parsing/alignment.

``step`` is the controlled collection protocol used in the paper
(``Reasoning: / Step i: / Conclusion:``). ``numbered``, ``bullet`` and ``free``
are the E5 robustness variants: the same dataset prompts with the step schema
rewritten. Every format keeps an explicit ``Conclusion:`` line because the
conclusion span is the anchor of the Stage-1 head.

Compared with the original ``data/claims.py`` path, the parser here keeps every
reasoning step (no hash-based trimming of the step list) and caps the count at
``MAX_STEPS``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from data.claims import _align_claims_to_tokens, _clean_claim_text, _extract_conclusion

MAX_STEPS = 12

STEP_CONTRACT = (
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

_STEP_LINE = re.compile(r"^(\s*)Step\s+(\d+)\s*:\s*", re.MULTILINE)


def _rewrite_step_lines(text: str, fmt: str) -> str:
    """Rewrite every ``Step k:`` line of a prompt into the target format."""
    if fmt == "step":
        return text
    if fmt == "numbered":
        return _STEP_LINE.sub(lambda m: f"{m.group(1)}{m.group(2)}. ", text)
    if fmt == "bullet":
        return _STEP_LINE.sub(lambda m: f"{m.group(1)}- ", text)
    if fmt == "free":
        # Merge consecutive step lines into one paragraph.
        lines = text.split("\n")
        out: List[str] = []
        buf: List[str] = []
        for line in lines:
            m = _STEP_LINE.match(line)
            if m:
                buf.append(line[m.end():].strip())
                continue
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(line)
        if buf:
            out.append(" ".join(buf))
        return "\n".join(out)
    raise ValueError(f"unknown format {fmt!r}")


def _rewrite_rules(text: str, fmt: str) -> str:
    if fmt == "step":
        return text
    repl = {
        "numbered": ("Step lines", "numbered lines"),
        "bullet": ("Step lines", "bullet lines"),
        "free": ("Step lines", "sentences"),
    }[fmt]
    text = text.replace(repl[0], repl[1]).replace("Step line", repl[1].rstrip("s"))
    if fmt == "bullet":
        text = text.replace("no markdown, code fences, bullets, or extra sections",
                            "no markdown, code fences, or extra sections")
        text = text.replace("No markdown, bullets, or code fences.", "No markdown or code fences.")
    if fmt == "free":
        text = text.replace("<one atomic fact>", "<your reasoning as a short paragraph>")
        text = text.replace("<one evidence statement>", "<your reasoning as a short paragraph>")
    return text


def build_messages(ds, raw_item: dict, fmt: str = "step") -> List[Dict[str, str]]:
    """Dataset messages + global output contract, rewritten for ``fmt``."""
    messages = [dict(m) for m in ds.build_chat_messages(raw_item)]
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = f'{msg["content"].rstrip()}\n\n{STEP_CONTRACT}'
            break
    else:
        messages.insert(0, {"role": "system", "content": STEP_CONTRACT})
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = _rewrite_rules(_rewrite_step_lines(msg["content"], fmt), fmt)
    return messages


def render_prompt_ids(tokenizer, messages, chat_template_kwargs: Optional[dict] = None) -> List[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    if hasattr(ids, "keys") and "input_ids" in ids.keys():  # BatchEncoding in transformers 5
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def _reasoning_section(text: str) -> Optional[str]:
    m = re.search(
        r"(?:^|\n)\s*\*{0,2}Reasoning\*{0,2}\s*:?\s*\n?(.*?)(?=(?:^|\n)\s*\*{0,2}(?:Conclusion|Final Answer)\*{0,2}\s*:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else None


def extract_steps(text: str, fmt: str) -> List[str]:
    section = _reasoning_section(text)
    if section is None:
        return []
    if fmt == "step":
        parts = re.findall(r"Step\s+\d+\s*:\s*(.+?)(?=Step\s+\d+\s*:|$)", section, re.DOTALL | re.IGNORECASE)
    elif fmt == "numbered":
        parts = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]\s|$)", section, re.DOTALL)
    elif fmt == "bullet":
        parts = re.findall(r"(?:^|\n)\s*[-*•]\s*(.+?)(?=\n\s*[-*•]\s|$)", section, re.DOTALL)
    elif fmt == "free":
        flat = section.strip()
        parts = [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", flat) if p.strip()]
    else:
        raise ValueError(fmt)
    steps = [_clean_claim_text(p) for p in parts]
    return [s for s in steps if s]


def parse_response(text: str, fmt: str, gen_token_ids: List[int], tokenizer) -> dict:
    """Parse steps + conclusion and align them to generated-token spans.

    Returns a dict with ``parse_ok`` and, when ok, ``steps`` and ``conclusion``
    entries of the form ``{"text": str, "tok": [start, end)}`` in generated-token
    coordinates.
    """
    steps = extract_steps(text, fmt)
    conclusion = _extract_conclusion(text)
    if not steps:
        return {"parse_ok": False, "drop_reason": "no_steps"}
    if not conclusion:
        return {"parse_ok": False, "drop_reason": "no_conclusion"}
    if len(steps) > MAX_STEPS:
        return {"parse_ok": False, "drop_reason": "too_many_steps"}

    claims = [{"text": s, "claim_type": "reasoning"} for s in steps]
    claims.append({"text": conclusion, "claim_type": "conclusion"})
    try:
        claims = _align_claims_to_tokens(claims, text, gen_token_ids, tokenizer)
    except ValueError as exc:
        return {"parse_ok": False, "drop_reason": f"align:{str(exc)[:80]}"}

    def _span(c):
        ids = c["aligned_token_ids"]
        return [int(min(ids)), int(max(ids)) + 1]

    return {
        "parse_ok": True,
        "drop_reason": "",
        "steps": [{"text": c["text"], "tok": _span(c)} for c in claims[:-1]],
        "conclusion": {"text": claims[-1]["text"], "tok": _span(claims[-1])},
    }
