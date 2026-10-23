"""MuSiQue (answerable): composed 2-4 hop questions over paragraphs with distractors."""

from chainuq.data.base import ReasoningDataset
from chainuq.data.prompts import build_system_prompt


class MuSiQue(ReasoningDataset):
    name, dir_name = "musique", "MuSiQue"
    context_char_budget = 8000
    system_prompt = build_system_prompt(
        objective="Solve the question by chaining sub-answers from the provided passages.",
        conclusion_schema="[final short answer]",
        examples=[{
            "input": ("Context:\n"
                      "[Film] Interstellar was directed by Christopher Nolan.\n"
                      "[Person] Christopher Nolan was born in London.\n"
                      "Question: In which city was the director of Interstellar born?"),
            "output": ("Reasoning:\n"
                       "Step 1: The director of Interstellar is Christopher Nolan.\n"
                       "Step 2: Christopher Nolan was born in London.\n"
                       "Conclusion: London."),
        }],
        extra_rules=["Answer sub-questions in order; each Step should carry one intermediate answer.",
                     "Conclusion must be only the final short answer, not a full sentence."],
    )

    def prepare(self, item):
        decomp = item.get("question_decomposition", item.get("decomposition", []))
        item["n_hops"] = max(len(decomp), 2) if isinstance(decomp, list) else 2

    def user_message(self, item):
        question = item.get("question", "")
        paragraphs = item.get("paragraphs", [])
        if not (paragraphs and isinstance(paragraphs, list)):
            return f"Question: {question}"
        blocks = [f"[{p.get('title', '')}] {p.get('paragraph_text', p.get('text', ''))}" if isinstance(p, dict) else p
                  for p in paragraphs if isinstance(p, (dict, str))]
        text, truncated = self.pack_context_blocks(blocks, max_chars=self.context_char_budget)
        if truncated:
            text += "\n\n[Note] Context truncated for length."
        return f"Context:\n{text}\n\nQuestion: {question}"

    def get_ground_truth(self, item):
        return str(item.get("answer", item.get("predicted_answer", "")))
