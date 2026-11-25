"""Judge-response parsing helpers shared by judge utilities."""
import json


def parse_judge_response_strict(response: str) -> dict | None:
    """Parse a single JSON object from the judge output.

    Strict mode only accepts one standalone JSON object with no prefix/suffix.
    Truncated JSON, arrays, and extra trailing text all fail (return ``None``).
    """
    raw = (response or "").strip()
    if not raw:
        return None
    if not raw.startswith("{"):
        return None
    try:
        parsed, end = json.JSONDecoder().raw_decode(raw, 0)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if raw[end:].strip():
        return None
    return parsed
