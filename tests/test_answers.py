"""Answer matching behind the judge-free labels (GSM8K, MATH) and the exact-match labels (StrategyQA)."""

import pytest

from chainuq.data.answers import extract_conclusion_line, math_equal, parse_yes_no_answer


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


def test_conclusion_line_and_yes_no():
    text = "Reasoning:\nStep 1: Penguins are birds.\nConclusion: Yes."
    assert extract_conclusion_line(text) == "Yes"
    assert parse_yes_no_answer(text) == "yes"
    assert parse_yes_no_answer("Reasoning:\nStep 1: x.\nConclusion: false") == "no"
