"""Final-answer extraction and answer matching (exact-match and math labels)."""

from __future__ import annotations

import re

_MARKER_PATTERNS = (
    r"(?:^|\n)\s*conclusion\s*[:\-]\s*([^\n\r]+)",
    r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*([^\n\r]+)",
    r"(?:^|\n)\s*answer\s*[:\-]\s*([^\n\r]+)",
)


def _clean_line(text: str) -> str:
    text = (text or "").strip().strip(" .,:;!?\"'")
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is|should\s+be)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:option|choice)\s+", "", text, flags=re.IGNORECASE)
    return text.strip().strip(" .,:;!?\"'")


def extract_conclusion_line(generated_text: str) -> str:
    """The last ``Conclusion:`` / ``Final answer:`` / ``Answer:`` line, else the last non-empty line."""
    text = (generated_text or "").strip()
    if not text:
        return ""
    for pattern in _MARKER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return _clean_line(matches[-1])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clean_line(lines[-1]) if lines else ""


def parse_yes_no_answer(generated_text: str) -> str:
    candidates = [extract_conclusion_line(generated_text)]
    full = (generated_text or "").strip()
    if full:
        candidates.append(full)
    for candidate in candidates:
        lower = candidate.lower()
        for pattern, normalized in ((r"\b(yes|true)\b", "yes"), (r"\b(no|false)\b", "no")):
            if re.findall(pattern, lower, flags=re.IGNORECASE):
                return normalized
    return "unknown"


def yes_no_text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value or "").strip().lower()
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"
    return text


def last_number(text: str) -> str:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return nums[-1].replace(",", "") if nums else ""


def last_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` (balanced braces), or ``""``."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    i = text.find("{", idx)
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1: j]
        j += 1
    return ""


def normalize_math(ans: str) -> str:
    s = str(ans or "").strip()
    if "\\boxed" in s:
        s = last_boxed(s) or s
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mbox|mathrm)\{\s*([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""),
                 ("\\$", ""), ("degrees", ""), ("units", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # "x = 5" -> "5"
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _to_float(s: str):
    s = s.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m and int(m.group(3)) != 0:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    return None


def math_equal(pred: str, gold: str) -> bool:
    """Normalized-string or numeric equality of a predicted and a reference math answer."""
    pred = str(pred or "")
    if "=" in pred and "\\boxed" not in pred:  # "2 + 5/3 + 1 = 14/3" -> "14/3"
        pred = pred.rsplit("=", 1)[1]
    a, b = normalize_math(pred), normalize_math(gold)
    if not a:
        return False
    if a == b:
        return True
    fa, fb = _to_float(a), _to_float(b)
    return fa is not None and fb is not None and abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
