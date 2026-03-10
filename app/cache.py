"""
In-memory caching for embeddings and document grades.
Reduces redundant OpenAI API calls for repeated queries and reformulations.
"""
import hashlib
from typing import Dict, Optional
import numpy as np


class EmbeddingCache:
    """LRU-like cache for text embeddings keyed by content hash."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, np.ndarray] = {}
        self._max_size = max_size

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._cache.get(self._key(text))

    def set(self, text: str, embedding: np.ndarray) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[self._key(text)] = embedding

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()


class GradeCache:
    """Cache batch grading results keyed by (question, all_doc_texts_hash)."""

    def __init__(self, max_size: int = 500):
        self._cache: Dict[str, list] = {}
        self._max_size = max_size

    def get(self, question: str, docs_fingerprint: str) -> Optional[list]:
        return self._cache.get(self._key(question, docs_fingerprint))

    def set(self, question: str, docs_fingerprint: str, grades: list) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[self._key(question, docs_fingerprint)] = grades

    def _key(self, question: str, docs_fingerprint: str) -> str:
        return hashlib.md5(f"{question}||{docs_fingerprint}".encode()).hexdigest()


# Global singletons — imported by vector_store.py and grader.py
embedding_cache = EmbeddingCache()
grade_cache = GradeCache()
