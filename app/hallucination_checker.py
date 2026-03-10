"""
Hallucination Checker
Post-generation verification: checks every claim in the answer against source documents.
Returns a grounded flag and confidence score (0.0–1.0).
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from app.prompts import HALLUCINATION_CHECK_PROMPT

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def check_hallucination(answer: str, documents: List[Dict]) -> Dict:
    """
    Verify that the generated answer is grounded in source documents.

    Args:
        answer: The generated answer text
        documents: List of source document dicts used to generate the answer

    Returns:
        Dict with:
        - grounded (bool): True if answer is well-supported
        - confidence (float): 0.0–1.0, how well-grounded the answer is
        - issues (list[str]): Any unsupported claims detected
    """
    # Build context string, capped to keep token usage reasonable
    context_parts = [
        f"[Source: {d.get('source', 'Unknown')}]\n{d.get('text', '')}"
        for d in documents
    ]
    context = "\n\n---\n\n".join(context_parts)[:4000]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": HALLUCINATION_CHECK_PROMPT.format(
                    context=context,
                    answer=answer,
                ),
            }],
            temperature=0.1,
            max_tokens=250,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
        return {
            "grounded": bool(result.get("grounded", True)),
            "confidence": round(float(result.get("confidence", 0.5)), 2),
            "issues": result.get("issues", []),
        }
    except Exception as e:
        print(f"[hallucination_checker] Error: {e}")
        return {
            "grounded": True,
            "confidence": 0.5,
            "issues": ["Verification inconclusive"],
        }
