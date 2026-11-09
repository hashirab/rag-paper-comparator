"""Turn retrieved chunks from both papers into a grounded, cited answer.

The point of this layer isn't "summarize the passages" - a plain RAG
answer would do that. It's to force the model to state, explicitly, where
the two papers' approaches agree, disagree, or where one simply doesn't
address something the other does - and to refuse rather than invent an
answer when the retrieved evidence doesn't actually cover the question.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

from retrieve import Hit

load_dotenv()

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a precise technical assistant comparing two research papers \
about retrieval-augmented generation. You are given passages retrieved from both \
papers, each tagged with its source citation and a kind (prose, equation, caption, \
or table_data).

Rules:
1. Use ONLY the passages given below. Do not use outside knowledge of these papers, \
even if you recognize them and recall their actual numbers - use what's on the page.
2. Every factual claim you make must end with its citation in square brackets, \
exactly as given, e.g. [RAG (Lewis et al., 2020), 2 Methods (p.3)].
3. If the question asks for a comparison, explicitly structure your answer around \
agreement, disagreement, and what one paper addresses that the other doesn't.
4. If the retrieved passages do not contain enough information to answer, say so \
plainly instead of guessing. Do not pad a thin answer with speculation.
5. Passages tagged (table_data) are PDF table rows flattened into a single line of \
text - the row/column grid is lost, so a number's position no longer reliably tells \
you which model or metric it belongs to. Only attribute a specific number to a \
specific model/column if the passage's wording unambiguously ties them together; \
otherwise describe the trend without inventing a precise attribution.
6. Be concise. No filler, no restating the question."""


def _format_passages(hits: list[Hit]) -> str:
    lines = []
    for h in hits:
        lines.append(f"[{h.citation()}] ({h.kind})\n{h.text}")
    return "\n\n".join(lines)


def answer(query: str, hits: list[Hit]) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())
    passages = _format_passages(hits)
    user_prompt = f"Question: {query}\n\nRetrieved passages:\n\n{passages}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content
