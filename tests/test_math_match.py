"""Answer matching for the judge-free math labels (GSM8K / MATH-500)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.math_reasoning import math_equal


@pytest.mark.parametrize("pred,gold,expected", [
    ("2 + 5/3 + 1 = (6 + 5 + 3)/3 = 14/3", r"\frac{14}{3}", True),
    (r"$\left( 3, \  \frac{\pi}{2}\right)$", r"\left( 3, \frac{\pi}{2} \right)", True),
    ("x = 5", "5", True),
    ("0.5", r"\frac{1}{2}", True),
    (r"\boxed{Angela}", r"\text{Evelyn}", False),
    ("1,000", "1000", True),
    (r"\dfrac{3}{4}", r"\frac34", True),
    ("7", "8", False),
    ("", "0", False),
])
def test_math_equal(pred, gold, expected):
    assert math_equal(pred, gold) is expected
