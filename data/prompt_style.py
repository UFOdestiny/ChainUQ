"""Unified prompt style helpers for dataset system prompts."""

from __future__ import annotations

from typing import Iterable, Sequence


def build_system_prompt(
    *,
    objective: str,
    conclusion_schema: str,
    examples: Sequence[dict] | None = None,
    max_steps: int = 6,
    max_step_words: int = 18,
    extra_rules: Iterable[str] | None = None,
) -> str:
    lines: list[str] = [
        "You are a careful reasoning assistant.",
        f"Task objective: {objective}",
        "",
        "Output schema (must follow exactly):",
        "Reasoning:",
        "Step 1: <one atomic fact>",
        "Step 2: <one atomic fact>",
        "...",
        f"Conclusion: {conclusion_schema}",
        "",
        "Formatting constraints:",
        "- Plain text only; no markdown, code fences, bullets, or extra sections.",
        f"- Use 1-{max_steps} Step lines; use only as many steps as the evidence requires.",
        f"- Keep each Step line <= {max_step_words} words.",
        "- The Conclusion line must contain only the final answer format requested above.",
        "- Do not output any text before 'Reasoning:' or after the Conclusion line.",
    ]
    for rule in extra_rules or []:
        lines.append(f"- {str(rule).strip()}")

    if examples:
        lines.append("")
        lines.append("Few-shot examples:")
        for idx, ex in enumerate(examples, start=1):
            sample_input = str(ex.get("input", "")).strip()
            sample_output = str(ex.get("output", "")).strip()
            if not sample_input or not sample_output:
                continue
            lines.extend(
                [
                    f"Example {idx}",
                    "Input:",
                    sample_input,
                    "Output:",
                    sample_output,
                ]
            )
    return "\n".join(lines)
