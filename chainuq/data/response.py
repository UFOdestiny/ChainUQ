"""Parse a response into reasoning steps and a conclusion, aligned to generated-token spans.

A response is kept only if it follows the output contract (``Reasoning:`` /
``Step i:`` / ``Conclusion:``) and every claim aligns exactly to a contiguous
span of generated tokens; other responses are dropped with a reason.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

MAX_STEPS = 12


def clean_claim_text(text: str) -> str:
    """Strip markers and boilerplate so a claim is a short atomic statement."""
    text = text.strip()
    text = re.sub(r"^\*{1,2}", "", text).strip()
    text = re.sub(r"\*{1,2}$", "", text).strip()
    text = re.sub(r"^[-*•.)\s]+", "", text)  # leading bullets / step numbers
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|should\s+be)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:option|choice)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" .")
    if len(text.split()) > 25:  # drop long parenthetical explanations
        text = re.sub(r"\s*\([^)]{20,}\)", "", text)
    if len(text.split()) > 35:  # keep the first sentence
        m = re.match(r"^(.+?[.!?])\s", text)
        if m:
            text = m.group(1)
    return text.strip()


def _reasoning_section(text: str) -> Optional[str]:
    m = re.search(
        r"(?:^|\n)\s*\*{0,2}Reasoning\*{0,2}\s*:?\s*\n?(.*?)(?=(?:^|\n)\s*\*{0,2}(?:Conclusion|Final Answer)\*{0,2}\s*:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else None


def extract_steps(text: str) -> List[str]:
    section = _reasoning_section(text)
    if section is None:
        return []
    parts = re.findall(r"Step\s+\d+\s*:\s*(.+?)(?=Step\s+\d+\s*:|$)", section, re.DOTALL | re.IGNORECASE)
    return [s for s in (clean_claim_text(p) for p in parts) if s]


def extract_conclusion(text: str) -> Optional[str]:
    """The conclusion claim; only explicit answer headers are accepted."""
    patterns = [
        r"(?:^|\n)\s*\*{0,2}Conclusion\*{0,2}\s*:?\s*(.+)$",
        r"(?:^|\n)\s*#{1,3}\s*Conclusion\s*:?\s*(.+)$",
        r"(?:^|\n)\s*Conclusion\s*:\s*(.+)$",
        r"(?:^|\n)\s*Final\s+Answer\s*[:\-]?\s*(.+)$",
        r"(?:^|\n)\s*Answer\s*[:\-]?\s*(.+)$",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            claim = clean_claim_text(matches[-1])
            if claim:
                return claim
    # Multi-line conclusion blocks (one sentence per line).
    multi = re.findall(r"(?:^|\n)\s*\*{0,2}Conclusion\*{0,2}\s*:\s*((?:[^\n]|\n(?!\s*\n))+?)(?=\n\s*\n|\Z)",
                       text, flags=re.IGNORECASE | re.DOTALL)
    if not multi:
        multi = re.findall(r"(?:^|\n)\s*Conclusion\s*:\s*((?:[^\n]|\n(?!\s*\n))+?)(?=\n\s*\n|\Z)",
                           text, flags=re.IGNORECASE | re.DOTALL)
    if multi:
        claim = clean_claim_text(multi[-1])
        if claim:
            return claim
    return None


def _typographic_normalize(text: str) -> str:
    """Map curly quotes and dashes to ASCII (length-preserving)."""
    if not text:
        return text
    return (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))


def _alignment_candidates(claim_text: str) -> List[str]:
    """Claim strings from strict to progressively relaxed, without duplicates."""
    base = (claim_text or "").strip()
    if not base:
        return []
    candidates = [base]
    no_fence = re.sub(r"\s*```[\w-]*\s*$", "", base).strip()  # trailing code fence
    if no_fence:
        candidates.append(no_fence)
    no_comment = re.sub(r"\s*//.*$", "", no_fence or base).strip()  # trailing inline comment
    if no_comment:
        candidates.append(no_comment)
    first_sentence = re.split(r"(?<=[.!?])\s+", no_comment or base, maxsplit=1)[0].strip()
    if first_sentence:
        candidates.append(first_sentence)
    tn = _typographic_normalize(base)
    if tn and tn != base:
        candidates.append(tn)
    return list(dict.fromkeys(candidates))


def _normalize_with_char_map(text: str) -> tuple[str, List[int]]:
    """Collapse whitespace runs to one space; map each output char to its source index."""
    out_chars: List[str] = []
    char_map: List[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out_chars.append(" ")
                char_map.append(i)
                in_ws = True
            continue
        in_ws = False
        out_chars.append(ch)
        char_map.append(i)
    return "".join(out_chars), char_map


def _find_claim_span(generated_text: str, claim_text: str, search_start: int) -> tuple[int, str]:
    """Character offset and matched text of a claim: exact match first, then relaxed fallbacks."""
    idx = generated_text.find(claim_text, search_start)
    if idx != -1:
        return idx, claim_text
    for candidate in _alignment_candidates(claim_text):
        idx = generated_text.find(candidate, search_start)
        if idx != -1:
            return idx, candidate
    # Whitespace-normalized match.
    norm_generated, char_map = _normalize_with_char_map(generated_text)
    norm_start = 0
    if 0 <= search_start < len(generated_text):
        norm_start = len(re.sub(r"\s+", " ", generated_text[:search_start]))
    elif search_start >= len(generated_text):
        norm_start = len(norm_generated)
    for candidate in _alignment_candidates(claim_text):
        norm_candidate = re.sub(r"\s+", " ", candidate).strip()
        if not norm_candidate:
            continue
        norm_idx = norm_generated.find(norm_candidate, norm_start)
        if norm_idx == -1:
            continue
        gen_start = char_map[norm_idx]
        gen_end = char_map[norm_idx + len(norm_candidate) - 1] + 1
        return gen_start, generated_text[gen_start:gen_end]
    # Typographic variants on both strings.
    nc, ng = _typographic_normalize(claim_text), _typographic_normalize(generated_text)
    if nc:
        idx = ng.find(nc, search_start)
        if idx != -1 and idx + len(nc) <= len(generated_text):
            return idx, generated_text[idx: idx + len(nc)]
    return -1, ""


def align_claims_to_tokens(claims: List[str], generated_text: str, token_ids: List[int], tokenizer) -> List[List[int]]:
    """Token indices covered by each claim, searched in generation order. Raises if a claim does not align."""
    positions, cum = [], 0
    for tid in token_ids:
        n = len(tokenizer.decode([tid], skip_special_tokens=False))
        positions.append((cum, cum + n))
        cum += n
    search_start, out = 0, []
    for claim in claims:
        idx, matched = _find_claim_span(generated_text, claim, search_start)
        if idx == -1:
            raise ValueError(f"Claim text not found exactly in generated text: {claim!r}")
        end = idx + len(matched)
        ids = [t for t, (a, b) in enumerate(positions) if b > idx and a < end]
        if not ids:
            raise ValueError(f"Claim did not align to any generated tokens: {claim!r}")
        out.append(ids)
        search_start = end
    return out


def parse_response(text: str, gen_token_ids: List[int], tokenizer) -> Dict:
    """``{"parse_ok": True, "steps": [{"text", "tok": [s, e)}], "conclusion": {...}}`` in generated-token
    coordinates, or ``{"parse_ok": False, "drop_reason": ...}``."""
    steps = extract_steps(text)
    conclusion = extract_conclusion(text)
    if not steps:
        return {"parse_ok": False, "drop_reason": "no_steps"}
    if not conclusion:
        return {"parse_ok": False, "drop_reason": "no_conclusion"}
    if len(steps) > MAX_STEPS:
        return {"parse_ok": False, "drop_reason": "too_many_steps"}
    claims = steps + [conclusion]
    try:
        spans = align_claims_to_tokens(claims, text, gen_token_ids, tokenizer)
    except ValueError as exc:
        return {"parse_ok": False, "drop_reason": f"align:{str(exc)[:80]}"}
    tok = [[min(ids), max(ids) + 1] for ids in spans]
    return {
        "parse_ok": True,
        "drop_reason": "",
        "steps": [{"text": c, "tok": t} for c, t in zip(steps, tok[:-1])],
        "conclusion": {"text": conclusion, "tok": tok[-1]},
    }
