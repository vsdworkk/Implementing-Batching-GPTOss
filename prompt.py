from __future__ import annotations

from datetime import date
from textwrap import dedent
from typing import Dict, List


def _strip(text: str) -> str:
  return dedent(text).strip()


CURRENT_DATE = date.today().isoformat()

system_prompt = _strip(
  f"""
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {CURRENT_DATE}
Reasoning: high
# Valid channels: analysis, commentary, final. Channel must be included for every message.
"""
)

developer_prompt = _strip(
  """
# Instructions
You are an experienced technical recruiter. Review each resume "about me" section for clarity, professionalism, specificity, and evidence-backed achievement.
Classify the section as "good" when it is professional, concrete, and demonstrates measurable impact. Classify as "bad" when it is vague, unprofessional, or lacks useful detail.

# Output Format
1. Think through the evaluation in the analysis channel.
2. In the final channel, return a compact JSON object that follows this schema exactly:
{
  "classification": "good" | "bad",
  "justification": "one sentence explaining the decision"
}
Do not include any additional keys or commentary in the final channel.
"""
)

user_prompt_template = _strip(
  """
You will be given a candidate's resume "about me" section.

<about_me>
{about_me}
</about_me>

Respond only after you have fully assessed the writing quality and hiring relevance.
"""
)


def build_messages(about_me: str) -> List[Dict[str, str]]:
  """
  Construct chat messages in the harmony format expected by GPT-OSS models.
  """

  user_prompt = user_prompt_template.format(about_me=about_me.strip())
  return [
    {"role": "system", "content": system_prompt},
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": user_prompt},
  ]


__all__ = [
  "system_prompt",
  "developer_prompt",
  "user_prompt_template",
  "build_messages",
]
