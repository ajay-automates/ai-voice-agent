"""
Query Reformulator — Async version.
Rewrites a failing query using different keywords so retrieval has a better chance.
Triggered when the initial retrieval yields too few relevant documents.
"""

import os
from openai import AsyncOpenAI
from app.prompts import REFORMULATOR_PROMPT

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


async def reformulate_query(original_question: str) -> str:
    """
    Reformulate a query that didn't retrieve enough relevant documents.

    Args:
        original_question: The original user question

    Returns:
        A reformulated version of the question, or original if reformulation fails.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": REFORMULATOR_PROMPT.format(question=original_question),
            }],
            temperature=0.5,
            max_tokens=100,
        )
        reformulated = response.choices[0].message.content.strip()
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        return reformulated or original_question
    except Exception as e:
        print(f"[reformulator] Error: {e}")
        return original_question
