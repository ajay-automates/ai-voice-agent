"""
Guardrails — Input validation, prompt injection detection, harmful content blocking.
Applied before every LLM call to protect the pipeline.
"""

from typing import Dict


def check_guardrails(text: str) -> Dict:
    """
    Check if user input should be allowed or blocked.

    Returns:
        {"allowed": bool, "reason": str}
    """
    text_lower = text.lower().strip()

    # Block empty / too short
    if not text_lower or len(text_lower) < 2:
        return {"allowed": False, "reason": "Input too short"}

    # Block prompt injection attempts
    injection_patterns = [
        "ignore previous", "ignore above", "forget your instructions",
        "you are now", "pretend you are", "act as",
        "system prompt", "reveal your prompt", "show me your instructions",
        "disregard", "override your", "new instructions", "jailbreak",
        "do anything now", "dan mode", "developer mode",
    ]
    for pattern in injection_patterns:
        if pattern in text_lower:
            return {"allowed": False, "reason": "Potential prompt injection detected"}

    # Block harmful content requests
    harmful_patterns = [
        "how to hack", "how to steal", "how to make a bomb",
        "how to make drugs", "exploit vulnerability", "bypass security",
        "how to kill", "how to hurt",
    ]
    for pattern in harmful_patterns:
        if pattern in text_lower:
            return {"allowed": False, "reason": "Harmful content request blocked"}

    return {"allowed": True, "reason": "OK"}
