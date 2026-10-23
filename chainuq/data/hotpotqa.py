"""HotpotQA (distractor setting): two-hop questions over ten Wikipedia paragraphs."""

from chainuq.data.base import ReasoningDataset
from chainuq.data.prompts import build_system_prompt


class HotpotQA(ReasoningDataset):
    name, dir_name = "hotpotqa", "hotpot_qa"
    context_char_budget = 8000
    system_prompt = build_system_prompt(
        objective="Answer the multi-hop question using only the provided context.",
        conclusion_schema="[final short answer]",
        examples=[{
            "input": ("Context:\n"
                      "[A] Ada wrote the first algorithm for Babbage's engine.\n"
                      "[B] Ada Lovelace collaborated with Charles Babbage.\n"
                      "Question: Who collaborated with Charles Babbage?"),
            "output": ("Reasoning:\n"
                       "Step 1: Passage B states Ada Lovelace collaborated with Charles Babbage.\n"
                       "Step 2: The asked collaborator is therefore Ada Lovelace.\n"
                       "Conclusion: Ada Lovelace."),
        }],
        extra_rules=["Each Step should cite one passage title when possible.",
                     "Conclusion must be only the answer span, not a full sentence."],
    )

    def prepare(self, item):
        item["n_hops"] = 2

    def user_message(self, item):
        context = item.get("context", {})
        blocks = [f"[{title}] {''.join(sents)}" for title, sents in zip(context.get("title", []), context.get("sentences", []))]
        text, truncated = self.pack_context_blocks(blocks, max_chars=self.context_char_budget)
        if truncated:
            text += "\n\n[Note] Context truncated for length."
        return f"Context:\n{text}\n\nQuestion: {item['question']}"
