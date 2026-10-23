"""GSM8K and MATH: math word problems whose labels come from answer matching (no judge)."""

import re

from chainuq.data.answers import extract_conclusion_line, last_number, math_equal
from chainuq.data.base import ReasoningDataset
from chainuq.data.prompts import build_system_prompt


class GSM8K(ReasoningDataset):
    name, dir_name = "gsm8k", "gsm8k"
    shuffle_subset = False
    system_prompt = build_system_prompt(
        objective="Solve the grade-school math word problem.",
        conclusion_schema="[final numeric answer only]",
        examples=[{
            "input": "Question: Tom has 3 boxes with 4 apples each. He eats 2 apples. How many apples are left?",
            "output": ("Reasoning:\nStep 1: Tom starts with 3 * 4 = 12 apples.\n"
                       "Step 2: After eating 2 apples he has 12 - 2 = 10 apples.\nConclusion: 10"),
        }],
        max_steps=8,
        extra_rules=["Conclusion must be a single number without units."],
    )

    def prepare(self, item):
        item["gold"] = item["answer"].split("####")[-1].strip().replace(",", "")
        item["n_hops"] = item["answer"].count("<<")  # calculator steps in the reference solution

    def user_message(self, item):
        return f"Question: {item['question']}"

    def parse_answer(self, generated_text):
        return last_number(extract_conclusion_line(generated_text))

    def check_correctness(self, generated, ground_truth):
        return int(math_equal(self.parse_answer(generated), ground_truth))

    def get_difficulty(self, item):
        return int(item.get("n_hops", 0))

    def get_ground_truth(self, item):
        return item["gold"]


class MATH(ReasoningDataset):
    name, dir_name = "math500", "math500"
    shuffle_subset = False
    system_prompt = build_system_prompt(
        objective="Solve the competition math problem.",
        conclusion_schema="[final answer only, in simplest form; LaTeX allowed]",
        examples=[{
            "input": "Question: What is the sum of the roots of x^2 - 5x + 6 = 0?",
            "output": ("Reasoning:\nStep 1: The roots of x^2 - 5x + 6 = 0 are 2 and 3.\n"
                       "Step 2: Their sum is 2 + 3 = 5.\nConclusion: 5"),
        }],
        max_steps=8,
        max_step_words=30,
        extra_rules=["Conclusion must contain only the final answer, without explanation."],
    )

    def prepare(self, item):
        item["question"] = item["problem"]
        level = re.findall(r"\d+", str(item.get("level", "0")))
        item["n_hops"] = int(level[0]) if level else 0  # MATH difficulty level 1-5

    def user_message(self, item):
        return f"Question: {item['problem']}"

    def check_correctness(self, generated, ground_truth):
        return int(math_equal(self.parse_answer(generated), ground_truth))

    def get_difficulty(self, item):
        return int(item.get("n_hops", 0))
