"""StrategyQA: yes/no questions with supporting facts; labels by exact match."""

import re

from chainuq.data.answers import parse_yes_no_answer, yes_no_text
from chainuq.data.base import ReasoningDataset
from chainuq.data.prompts import build_system_prompt


def _fact_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    facts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return facts or [text]


class StrategyQA(ReasoningDataset):
    name, dir_name = "strategyqa", "StrategyQA"
    context_char_budget = 5000
    system_prompt = build_system_prompt(
        objective="Use the provided facts to answer the yes/no question.",
        conclusion_schema="[yes or no]",
        examples=[{
            "input": "Facts:\n- Penguins are birds.\n- Birds are animals.\nQuestion: Are penguins animals?",
            "output": ("Reasoning:\n"
                       "Step 1: Penguins are birds according to the facts.\n"
                       "Step 2: Birds are animals according to the facts.\n"
                       "Conclusion: yes."),
        }],
        extra_rules=["Use the facts only; do not rely on outside knowledge.",
                     "Conclusion must be exactly 'yes' or 'no'."],
    )

    def prepare(self, item):
        decomposition = item.get("decomposition") or []
        item["facts"] = _fact_list(item.get("facts"))
        n_hops = len(decomposition) if isinstance(decomposition, list) and decomposition else len(item["facts"])
        item["answer"] = yes_no_text(item.get("answer"))
        item["n_hops"] = max(2, min(5, n_hops or 2))

    def user_message(self, item):
        blocks = []
        term, description = str(item.get("term", "")).strip(), str(item.get("description", "")).strip()
        if term and description:
            blocks.append(f"[Topic] {term}: {description}")
        blocks += [f"- {str(f).strip()}" for f in item.get("facts") or [] if str(f).strip()]
        text, truncated = self.pack_context_blocks(blocks, max_chars=self.context_char_budget)
        if truncated:
            text += "\n\n[Note] Facts truncated for length."
        return f"Facts:\n{text}\n\nQuestion: {item.get('question', '')}"

    def parse_answer(self, generated_text):
        return yes_no_text(parse_yes_no_answer(generated_text))

    def check_correctness(self, generated, ground_truth):
        pred = yes_no_text(self.parse_answer(generated))
        return int(bool(pred) and pred == yes_no_text(ground_truth))

    def get_ground_truth(self, item):
        return yes_no_text(item.get("answer"))
