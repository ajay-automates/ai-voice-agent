"""
Document Relevance Grader
Uses GPT-4o-mini to evaluate whether each retrieved document actually answers the user's question.
This is the key step that makes agentic RAG smarter than basic RAG — embedding similarity != relevance.
"""

import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from app.prompts import GRADER_PROMPT

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def grade_documents(question: str, documents: List[Dict]) -> List[Dict]:
    """
    Grade each retrieved document for relevance to the question.

    Args:
        question: User's question
        documents: List of dicts with at least 'text' and 'source' keys

    Returns:
        Same list with 'grade' and 'grade_reason' added to each doc.
        grade values: "relevant" | "partially_relevant" | "not_relevant"
    """
    graded = []
    for doc in documents:
        doc_text = doc.get("text", "")[:1200]  # cap tokens per doc
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": GRADER_PROMPT.format(
                        document=doc_text,
                        question=question,
                    ),
                }],
                temperature=0.1,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].replace("json", "").strip()
            result = json.loads(raw)
            doc = dict(doc)  # copy so we don't mutate caller's list
            doc["grade"] = result.get("relevance", "partially_relevant")
            doc["grade_reason"] = result.get("reason", "")
        except Exception as e:
            doc = dict(doc)
            doc["grade"] = "partially_relevant"
            doc["grade_reason"] = f"Grading error: {str(e)}"
        graded.append(doc)
    return graded


def count_relevant_documents(graded_docs: List[Dict]) -> Tuple[int, int, int]:
    """
    Count documents by grade.
    Returns (relevant_count, partially_relevant_count, not_relevant_count).
    """
    relevant = sum(1 for d in graded_docs if d.get("grade") == "relevant")
    partial = sum(1 for d in graded_docs if d.get("grade") == "partially_relevant")
    not_rel = sum(1 for d in graded_docs if d.get("grade") == "not_relevant")
    return relevant, partial, not_rel
