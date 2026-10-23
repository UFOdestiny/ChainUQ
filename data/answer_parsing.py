"""Shared answer parsing helpers with stricter fallback behavior."""

from __future__ import annotations

import re

_MARKER_PATTERNS = (
    r"(?:^|\n)\s*conclusion\s*[:\-]\s*([^\n\r]+)",
    r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*([^\n\r]+)",
    r"(?:^|\n)\s*answer\s*[:\-]\s*([^\n\r]+)",
)


def _clean_line(text: str) -> str:
    text = (text or "").strip().strip(" .,:;!?\"'")
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|should\s+be)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:option|choice)\s+", "", text, flags=re.IGNORECASE)
    return text.strip().strip(" .,:;!?\"'")


def extract_conclusion_line(generated_text: str) -> str:
    text = (generated_text or "").strip()
    if not text:
        return ""
    for pattern in _MARKER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return _clean_line(matches[-1])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clean_line(lines[-1]) if lines else ""


def parse_free_form_answer(generated_text: str) -> str:
    return extract_conclusion_line(generated_text)


def parse_yes_no_answer(generated_text: str) -> str:
    candidates = [extract_conclusion_line(generated_text)]
    full = (generated_text or "").strip()
    if full:
        candidates.append(full)

    for candidate in candidates:
        lower = candidate.lower()
        for pattern, normalized in (
            (r"\b(yes|true)\b", "yes"),
            (r"\b(no|false)\b", "no"),
        ):
            matches = re.findall(pattern, lower, flags=re.IGNORECASE)
            if matches:
                return normalized
    return "unknown"


def parse_choice_index(
    generated_text: str,
    *,
    num_options: int,
    allow_letters: bool = True,
) -> str:
    if num_options <= 0:
        return "unknown"

    candidates = [extract_conclusion_line(generated_text)]
    text = (generated_text or "").strip()
    if text:
        # Only admit broader fallback when output explicitly looks like answer text.
        lower = text.lower()
        if "answer" in lower or "conclusion" in lower or "option" in lower:
            candidates.append(text)

    letter_to_index = {chr(ord("a") + i): i for i in range(min(num_options, 26))}
    max_digit = min(num_options, 9)

    for candidate in candidates:
        lower = candidate.lower()
        explicit = re.findall(
            rf"(?:answer(?:\s+is)?|option|choice)\s*[:\-]?\s*([1-{max_digit}])\b",
            lower,
            flags=re.IGNORECASE,
        )
        if explicit:
            return str(int(explicit[-1]) - 1)
        direct_digit = re.findall(rf"\b([1-{max_digit}])\b", lower)
        if direct_digit:
            return str(int(direct_digit[-1]) - 1)

        if allow_letters and letter_to_index:
            explicit_letter = re.findall(
                r"(?:answer(?:\s+is)?|option|choice)\s*[:\-]?\s*([a-z])\b",
                lower,
                flags=re.IGNORECASE,
            )
            if explicit_letter and explicit_letter[-1] in letter_to_index:
                return str(letter_to_index[explicit_letter[-1]])
            direct_letter = re.findall(r"\b([a-z])\b", lower, flags=re.IGNORECASE)
            if direct_letter and direct_letter[-1] in letter_to_index:
                return str(letter_to_index[direct_letter[-1]])

    return "unknown"
