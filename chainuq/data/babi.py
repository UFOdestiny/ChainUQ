"""bAbI QA (20 tasks): entity and event tracking over short synthetic passages."""

from chainuq.data.base import ReasoningDataset
from chainuq.data.prompts import build_system_prompt

# Approximate number of supporting facts (reasoning hops) per bAbI task.
TASK_TO_HOPS = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 2, 10: 1,
                11: 2, 12: 1, 13: 2, 14: 2, 15: 1, 16: 2, 17: 2, 18: 1, 19: 3, 20: 3}


class BAbI(ReasoningDataset):
    name, dir_name = "babi", "babi"
    system_prompt = build_system_prompt(
        objective="Answer the question by tracking entities and events in the passage.",
        conclusion_schema="[final short answer]",
        examples=[{
            "input": "Passage:\nMary went to the kitchen. John moved to the garden.\nQuestion: Where is John?",
            "output": ("Reasoning:\n"
                       "Step 1: The passage says John moved to the garden.\n"
                       "Step 2: No later sentence changes John's location.\n"
                       "Conclusion: garden."),
        }],
        extra_rules=["Track the latest relevant entity state; ignore earlier states overwritten later.",
                     "Conclusion must be only the requested entity, location, object, or yes/no answer."],
    )

    def prepare(self, item):
        task = item.get("task", 1)
        if isinstance(task, str):
            task = int(task) if task.isdigit() else 1
        item["task"] = task
        item["n_hops"] = TASK_TO_HOPS.get(task, 2)

    def user_message(self, item):
        passage = item["passage"]
        if isinstance(passage, list):
            passage = "\n".join(passage)
        return f"Passage:\n{passage}\n\nQuestion: {item['question']}"
