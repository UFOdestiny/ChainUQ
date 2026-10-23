"""Response parsing and alignment of claims to generated-token spans."""

from chainuq.data.response import extract_conclusion, extract_steps, parse_response


class CharTokenizer:
    """One token per character: token spans equal character spans."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


TEXT = "Reasoning:\nStep 1: Ada met Babbage.\nStep 2: They built an engine.\nConclusion: Ada Lovelace."


def test_steps_and_conclusion():
    assert extract_steps(TEXT) == ["Ada met Babbage", "They built an engine"]
    assert extract_conclusion(TEXT) == "Ada Lovelace"


def test_parse_response_spans():
    out = parse_response(TEXT, [ord(c) for c in TEXT], CharTokenizer())
    assert out["parse_ok"]
    for claim in out["steps"] + [out["conclusion"]]:
        a, b = claim["tok"]
        assert TEXT[a:b] == claim["text"]


def test_parse_failures():
    tok = CharTokenizer()
    assert parse_response("Conclusion: x", [ord(c) for c in "Conclusion: x"], tok)["drop_reason"] == "no_steps"
    text = "Reasoning:\nStep 1: a fact."
    assert parse_response(text, [ord(c) for c in text], tok)["drop_reason"] == "no_conclusion"
