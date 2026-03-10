"""
Document Relevance Grader — Batch version.
Grades ALL retrieved documents in ONE LLM call instead of 1 call per doc.
Saves ~3 seconds vs the sequential approach (5 docs × ~600ms each).
"""

import json
import os
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
from langsmith import traceable

from app.cache import grade_cache

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


@traceable(name="grade_documents_batch", run_type="llm")
async def grade_documents(question: str, documents: List[Dict]) -> List[Dict]:
    """
    Grade ALL documents in a single LLM call.

    Args:
        question: User's question
        documents: List of dicts with at least 'text' and 'source' keys

    Returns:
        Same list with 'grade' and 'grade_reason' added to each doc.
        grade values: "relevant" | "partially_relevant" | "not_relevant"
    """
    if not documents:
        return []

    # Cache key: question + fingerprint of all doc texts
    docs_fingerprint = "".join(d.get("text", "")[:80] for d in documents)
    cached = grade_cache.get(question, docs_fingerprint)
    if cached is not None:
        return cached

    # Build a single prompt with all docs
    docs_text = ""
    for i, doc in enumerate(documents):
        doc_text = doc.get("text", "")[:1000]
        docs_text += f"[Doc {i + 1}]:\n{doc_text}\n\n"

    batch_prompt = f"""Grade each document for relevance to the question.

Question: {question}

Documents:
{docs_text}
Respond with ONLY a JSON array (one entry per doc, in order), no other text:
[
  {{"relevance": "relevant|partially_relevant|not_relevant", "reason": "one sentence"}},
  ...
]"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        grades = json.loads(raw)

        graded = []
        for i, doc in enumerate(documents):
            d = dict(doc)
            if i < len(grades):
                d["grade"] = grades[i].get("relevance", "partially_relevant")
                d["grade_reason"] = grades[i].get("reason", "")
            else:
                d["grade"] = "partially_relevant"
                d["grade_reason"] = "Grading incomplete"
            graded.append(d)

        grade_cache.set(question, docs_fingerprint, graded)
        return graded

    except Exception as e:
        # Fallback: mark all as partially_relevant so pipeline continues
        return [
            {**dict(doc), "grade": "partially_relevant", "grade_reason": f"Grading error: {str(e)[:50]}"}
            for doc in documents
        ]


def count_relevant_documents(graded_docs: List[Dict]) -> Tuple[int, int, int]:
    """
    Count documents by grade.
    Returns (relevant_count, partially_relevant_count, not_relevant_count).
    """
    relevant = sum(1 for d in graded_docs if d.get("grade") == "relevant")
    partial = sum(1 for d in graded_docs if d.get("grade") == "partially_relevant")
    not_rel = sum(1 for d in graded_docs if d.get("grade") == "not_relevant")
    return relevant, partial, not_rel
