"""Claim extraction and alignment for multi-hop reasoning chains.

Strict mode: only regex-structured claims are accepted, and every claim must
align exactly to generated-token spans. Samples that fail this contract are
rejected upstream instead of being weakly recovered with fallbacks.
"""
import re
import hashlib
from typing import Dict, List, Optional
from utils.log import get_logger

log = get_logger(__name__)

# Claim type IDs for embedding
CLAIM_TYPE_IDS = {
    "reasoning": 0,
    "conclusion": 1,
    "unknown": 2,
}
MAX_REASONING_CLAIMS = 9
FIXED_HOP_DATASETS = {"spartqa", "hotpotqa", "hotpot_qa"}


def make_claim_stable_key(
    *,
    dataset_name: Optional[str] = None,
    question: Optional[str] = None,
    n_hops: Optional[int] = None,
    generated_text: Optional[str] = None,
) -> str:
    """Build a deterministic claim-trimming key without ground-truth leakage."""
    parts = [
        (dataset_name or "").strip().lower(),
        str(n_hops if n_hops is not None else ""),
        (question or "").strip(),
        (generated_text or "").strip(),
    ]
    return "|".join(parts)


def _normalize_hops(n_hops: Optional[int]) -> int:
    if n_hops is None:
        return MAX_REASONING_CLAIMS
    try:
        hops = int(n_hops)
    except (TypeError, ValueError):
        return MAX_REASONING_CLAIMS
    if hops <= 0:
        return MAX_REASONING_CLAIMS
    return hops


def _is_fixed_hop_dataset(dataset_name: Optional[str]) -> bool:
    name = (dataset_name or "").strip().lower()
    return name in FIXED_HOP_DATASETS


def reasoning_claim_bounds(
    n_hops: Optional[int],
    dataset_name: Optional[str] = None,
) -> tuple[int, int]:
    """Return allowed [low, high] range for reasoning claim count."""
    hops = _normalize_hops(n_hops)
    if _is_fixed_hop_dataset(dataset_name):
        low = max(1, min(MAX_REASONING_CLAIMS, hops))
        high = max(low, min(MAX_REASONING_CLAIMS, hops + 4))
        return low, high
    low = max(1, min(MAX_REASONING_CLAIMS, hops - 1))
    high = max(low, min(MAX_REASONING_CLAIMS, hops + 1))
    return low, high


def trim_reasoning_claims(
    reasoning_claims: List[str] | List[dict],
    n_hops: Optional[int],
    dataset_name: Optional[str] = None,
    stable_key: Optional[str] = None,
):
    """Trim reasoning claims with deterministic jitter inside allowed range."""
    if not reasoning_claims:
        return reasoning_claims
    low, high = reasoning_claim_bounds(n_hops, dataset_name=dataset_name)
    high = min(high, len(reasoning_claims))
    if len(reasoning_claims) <= low:
        return reasoning_claims
    if high <= low:
        return reasoning_claims[:high]

    if stable_key:
        digest = hashlib.blake2b(str(stable_key).encode("utf-8"), digest_size=4).digest()
        pick = int.from_bytes(digest, "little")
        target = low + (pick % (high - low + 1))
    else:
        target = high
    return reasoning_claims[:target]


def extract_claims_from_generation(
    generated_text: str,
    generated_token_ids: list,
    tokenizer=None,
    n_hops: Optional[int] = None,
    dataset_name: Optional[str] = None,
    stable_key: Optional[str] = None,
) -> List[Dict]:
    """Extract claims from structured reasoning output.

    Expected format:
        Reasoning:
        Step 1: [claim text]
        Step 2: [claim text]
        ...
        Conclusion: [final answer]

    Args:
        generated_text: Full generated text from LLM
        generated_token_ids: Token IDs of generated text
        tokenizer: Tokenizer for token alignment

    Returns:
        List of claim dicts with keys:
            - text: claim text
            - claim_type: "reasoning" or "conclusion"
            - claim_type_id: numeric type ID
            - aligned_token_ids: list of token indices
            - verified: -1 (pending verification)
    """
    reasoning_claims = _extract_reasoning_steps(generated_text)
    conclusion_claim = _extract_conclusion(generated_text)
    if not reasoning_claims:
        raise ValueError("Missing structured reasoning steps for strict regex extraction.")
    if not conclusion_claim:
        raise ValueError("Missing explicit conclusion for strict regex extraction.")

    # Hard cardinality contract: <=cap(n_hops, dataset) reasoning, exactly 1 conclusion.
    reasoning_claims = trim_reasoning_claims(
        reasoning_claims,
        n_hops=n_hops,
        dataset_name=dataset_name,
        stable_key=stable_key,
    )
    if not reasoning_claims:
        raise ValueError("No reasoning claims remain after strict trimming.")

    claims = []
    # Build claims with type info
    for claim_text in reasoning_claims:
        claims.append({
            "text": claim_text,
            "claim_type": "reasoning",
            "claim_type_id": CLAIM_TYPE_IDS["reasoning"],
            "aligned_token_ids": [],
            "verified": -1,
        })

    claims.append({
        "text": conclusion_claim,
        "claim_type": "conclusion",
        "claim_type_id": CLAIM_TYPE_IDS["conclusion"],
        "aligned_token_ids": [],
        "verified": -1,
    })

    # Align claims to token positions
    if tokenizer is None or generated_token_ids is None:
        raise ValueError("Strict claim extraction requires generated token ids and tokenizer.")
    claims = _align_claims_to_tokens(claims, generated_text, generated_token_ids, tokenizer)

    return claims


def _typographic_normalize(text: str) -> str:
    """Normalize curly quotes / dashes so regex-extracted claims match generation."""
    if not text:
        return text
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _extract_reasoning_steps(text: str) -> List[str]:
    """Extract strict ``Step N:`` lines from the ``Reasoning:`` block only."""
    steps = []

    # Strict contract: a dedicated Reasoning section must exist.
    reasoning_match = re.search(
        r'(?:^|\n)\s*\*{0,2}Reasoning\*{0,2}\s*:?\s*\n(.*?)(?=Conclusion:|Final Answer:|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not reasoning_match:
        return steps
    section = reasoning_match.group(1)

    step_matches = re.findall(
        r'Step\s+\d+\s*:\s*(.+?)(?=Step\s+\d+\s*:|Conclusion:|Final Answer:|$)',
        section,
        re.DOTALL | re.IGNORECASE,
    )
    for match in step_matches:
        claim = _clean_claim_text(match)
        if claim:
            steps.append(claim)

    return steps


def _extract_conclusion(text: str) -> Optional[str]:
    """Extract conclusion claim from text."""
    # Strict contract: only explicit answer headers are accepted.
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
            claim = _clean_claim_text(matches[-1])
            if claim:
                return claim
    # Multi-line conclusion blocks (common when models add one sentence per line).
    multi = re.findall(
        r"(?:^|\n)\s*\*{0,2}Conclusion\*{0,2}\s*:\s*((?:[^\n]|\n(?!\s*\n))+?)(?=\n\s*\n|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not multi:
        multi = re.findall(
            r"(?:^|\n)\s*Conclusion\s*:\s*((?:[^\n]|\n(?!\s*\n))+?)(?=\n\s*\n|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if multi:
        claim = _clean_claim_text(multi[-1])
        if claim:
            return claim
    return None


def _clean_claim_text(text: str) -> str:
    """Clean a claim text string into a short atomic statement."""
    text = text.strip()
    text = re.sub(r"^\*{1,2}", "", text).strip()
    text = re.sub(r"\*{1,2}$", "", text).strip()
    # Remove leading markers (bullets, step numbers)
    text = re.sub(r'^[-*\u2022.)\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' .')
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|should\s+be)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:option|choice)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(' .')
    # Remove parenthetical explanations if claim is long
    if len(text.split()) > 25:
        text = re.sub(r'\s*\([^)]{20,}\)', '', text)
    # If still very long, take up to the first sentence-ending punctuation
    if len(text.split()) > 35:
        m = re.match(r'^(.+?[.!?])\s', text)
        if m:
            text = m.group(1)
    return text.strip()


def _align_claims_to_tokens(
    claims: List[Dict],
    generated_text: str,
    generated_token_ids,
    tokenizer,
) -> List[Dict]:
    """Align each claim to its corresponding token positions."""
    if hasattr(generated_token_ids, 'tolist'):
        token_ids_list = generated_token_ids.tolist()
    else:
        token_ids_list = list(generated_token_ids)

    # Decode each token to build position map
    token_texts = []
    for tid in token_ids_list:
        decoded = tokenizer.decode([tid], skip_special_tokens=False)
        token_texts.append(decoded)

    # Build cumulative text position map
    cum_text = ""
    token_positions = []  # (start_char, end_char) for each token
    for t_text in token_texts:
        start = len(cum_text)
        cum_text += t_text
        end = len(cum_text)
        token_positions.append((start, end))

    search_start = 0
    # For each claim, find matching token range in generation order
    for claim in claims:
        claim_text = claim["text"]
        idx, matched_text = _find_claim_span(generated_text, claim_text, search_start)
        if idx == -1:
            raise ValueError(f"Claim text not found exactly in generated text: {claim_text!r}")

        end_idx = idx + len(matched_text)
        aligned_ids = []
        for t_idx, (t_start, t_end) in enumerate(token_positions):
            if t_end > idx and t_start < end_idx:
                aligned_ids.append(t_idx)
        if not aligned_ids:
            raise ValueError(f"Claim did not align to any generated tokens: {claim_text!r}")
        claim["aligned_token_ids"] = aligned_ids
        search_start = end_idx

    return claims


def _find_claim_span(generated_text: str, claim_text: str, search_start: int) -> tuple[int, str]:
    """Find claim span with exact match first, then robust fallbacks."""
    # 1) Exact match on extracted claim text.
    idx = generated_text.find(claim_text, search_start)
    if idx != -1:
        return idx, claim_text

    # 2) Exact match on progressively cleaned variants (code fence/comment tails).
    for candidate in _claim_alignment_candidates(claim_text):
        if not candidate:
            continue
        idx = generated_text.find(candidate, search_start)
        if idx != -1:
            return idx, candidate

    # 3) Whitespace-normalized fallback to tolerate newlines/tabs differences.
    norm_generated, char_map = _normalize_with_char_map(generated_text)
    norm_start = 0
    if 0 <= search_start < len(generated_text):
        norm_start = len(re.sub(r"\s+", " ", generated_text[:search_start]))
    elif search_start >= len(generated_text):
        norm_start = len(norm_generated)

    for candidate in _claim_alignment_candidates(claim_text):
        norm_candidate = re.sub(r"\s+", " ", candidate).strip()
        if not norm_candidate:
            continue
        norm_idx = norm_generated.find(norm_candidate, norm_start)
        if norm_idx == -1:
            continue
        gen_start = char_map[norm_idx]
        norm_end = norm_idx + len(norm_candidate) - 1
        gen_end = char_map[norm_end] + 1
        return gen_start, generated_text[gen_start:gen_end]

    # Typographic variants (curly vs ASCII quotes) — length-preserving map on both strings.
    nc = _typographic_normalize(claim_text)
    ng = _typographic_normalize(generated_text)
    if nc:
        idx = ng.find(nc, search_start)
        if idx != -1 and idx + len(nc) <= len(generated_text):
            return idx, generated_text[idx : idx + len(nc)]

    return -1, ""


def _claim_alignment_candidates(claim_text: str) -> List[str]:
    """Return candidate claim strings from strict to progressively relaxed."""
    base = (claim_text or "").strip()
    if not base:
        return []
    candidates = [base]

    # Trim markdown code-fence suffixes like ``` or ```python.
    no_fence = re.sub(r"\s*```[\w-]*\s*$", "", base).strip()
    if no_fence:
        candidates.append(no_fence)

    # Trim inline comments often emitted by instruction-tuned models.
    no_comment = re.sub(r"\s*//.*$", "", no_fence or base).strip()
    if no_comment:
        candidates.append(no_comment)

    # Keep the first sentence when the model appends extra commentary.
    first_sentence = re.split(r"(?<=[.!?])\s+", no_comment or base, maxsplit=1)[0].strip()
    if first_sentence:
        candidates.append(first_sentence)

    tn = _typographic_normalize(base)
    if tn and tn != base:
        candidates.append(tn)

    # De-duplicate while preserving order.
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _normalize_with_char_map(text: str) -> tuple[str, List[int]]:
    """Normalize whitespace and keep map from normalized chars to source indices."""
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
