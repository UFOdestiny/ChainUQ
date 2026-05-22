"""Number parsing utilities."""
import re


def first_number(*values, default=None):
    """Return the first numeric value found among *values*, or *default*.

    Each value may be a number (int/float), a string containing a number,
    or ``None``.  The function tries values left-to-right, returning the
    first one that yields a valid number.
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        m = re.search(r'-?\d+\.?\d*', str(v))
        if m:
            return float(m.group())
    return default


def to_float(value, default=None):
    """Safe float conversion with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
