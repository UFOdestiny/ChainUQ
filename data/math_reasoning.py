"""GSM8K and MATH-500: math-reasoning OOD benchmarks with judge-free labels.

Correctness is decided by answer matching (numeric equality for GSM8K;
normalised-string or numeric equality of the final answer for MATH), so these
two datasets give response-level labels that do not depend on any LLM judge.
"""

from __future__ import annotations

import re

from datasets import DatasetDict, load_from_disk

from data.base import BaseReasoningDataset
from data.answer_parsing import extract_conclusion_line
from data.prompt_style import build_system_prompt
from utils.log import get_logger

log = get_logger(__name__)


def last_boxed(text: str) -> str:
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
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if m:
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


def _last_number(text: str) -> str:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return nums[-1].replace(",", "") if nums else ""


class GSM8KDataset(BaseReasoningDataset):
    name = "gsm8k"
    task_type = "free_form"
    needs_judge = False
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

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path
        super().__init__(split=split, max_samples=max_samples, n_hop_values=n_hop_values, cache_dir=cache_dir)

    def _load_data(self):
        ds = load_from_disk(self.dataset_path)
        assert isinstance(ds, DatasetDict)
        items = [dict(x) for x in ds["validation" if self.split in ("val", "validation") else self.split]]
        for it in items:
            it["gold"] = it["answer"].split("####")[-1].strip().replace(",", "")
            it["n_hops"] = it["answer"].count("<<")  # number of calculator steps in the reference solution
        if self.max_samples > 0:
            items = items[: self.max_samples]
        return items

    def build_chat_messages(self, raw_item):
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {raw_item['question']}"}]

    def parse_answer(self, generated_text):
        return _last_number(extract_conclusion_line(generated_text))

    def check_correctness(self, generated, ground_truth):
        return int(math_equal(self.parse_answer(generated), ground_truth))

    def get_difficulty(self, raw_item):
        return int(raw_item.get("n_hops", 0))

    def get_ground_truth(self, raw_item):
        return raw_item["gold"]


class MATHDataset(BaseReasoningDataset):
    name = "math500"
    task_type = "free_form"
    needs_judge = False
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

    def __init__(self, split="train", max_samples=0, n_hop_values=None, cache_dir=None, dataset_path=None):
        self.dataset_path = dataset_path
        super().__init__(split=split, max_samples=max_samples, n_hop_values=n_hop_values, cache_dir=cache_dir)

    def _load_data(self):
        ds = load_from_disk(self.dataset_path)
        assert isinstance(ds, DatasetDict)
        items = [dict(x) for x in ds["validation" if self.split in ("val", "validation") else self.split]]
        for it in items:
            it["question"] = it["problem"]
            lvl = re.findall(r"\d+", str(it.get("level", "0")))
            it["n_hops"] = int(lvl[0]) if lvl else 0  # MATH difficulty level 1-5
        if self.max_samples > 0:
            items = items[: self.max_samples]
        return items

    def build_chat_messages(self, raw_item):
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {raw_item['problem']}"}]

    def parse_answer(self, generated_text):
        return extract_conclusion_line(generated_text)

    def check_correctness(self, generated, ground_truth):
        return int(math_equal(self.parse_answer(generated), ground_truth))

    def get_difficulty(self, raw_item):
        return int(raw_item.get("n_hops", 0))

    def get_ground_truth(self, raw_item):
        return str(raw_item["answer"])
