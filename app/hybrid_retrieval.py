"""
Hybrid Retrieval — BM25 keyword search + vector similarity, fused via Reciprocal Rank Fusion.
Combines semantic understanding (vector) with exact term matching (BM25).
"""

import re
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Searcher:
    """BM25 keyword-based search over the document store."""

    def __init__(self):
        self.bm25 = None
        self.documents: List[Dict] = []

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip special chars, split into tokens."""
        text = text.lower()
        text = re.sub(r'[^\w\s\.\$\%]', ' ', text)
        return [t for t in text.split() if len(t) > 1 or t.isdigit()]

    def build_index(self, documents: List[Dict]) -> None:
        """Build BM25 index from a list of {"text", "source"} dicts."""
        self.documents = documents
        tokenized = [self._tokenize(d.get("text", "")) for d in documents]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Return top-n BM25 results for query."""
        if not self.bm25 or not self.documents:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:n_results]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.documents[idx].get("text", ""),
                    "source": self.documents[idx].get("source", ""),
                    "bm25_score": float(scores[idx]),
                    "relevance_score": float(scores[idx]),
                    "retrieval_method": "bm25",
                })
        return results


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Merge vector + BM25 rankings using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across both lists.
    """
    fused: Dict[str, Dict] = {}

    for rank, doc in enumerate(vector_results):
        key = doc.get("text", "")[:120]
        if key not in fused:
            fused[key] = {
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "relevance_score": doc.get("relevance_score", 0.0),
            }
        fused[key]["rrf_score"] += 1.0 / (k + rank + 1)
        fused[key]["vector_rank"] = rank + 1

    for rank, doc in enumerate(bm25_results):
        key = doc.get("text", "")[:120]
        if key not in fused:
            fused[key] = {
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "relevance_score": doc.get("bm25_score", 0.0),
            }
        fused[key]["rrf_score"] += 1.0 / (k + rank + 1)
        fused[key]["bm25_rank"] = rank + 1

    merged = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    for doc in merged:
        if doc["vector_rank"] and doc["bm25_rank"]:
            doc["retrieval_method"] = "hybrid"
        elif doc["vector_rank"]:
            doc["retrieval_method"] = "vector"
        else:
            doc["retrieval_method"] = "bm25"

    return merged


def hybrid_search(
    query: str,
    vector_results: List[Dict],
    bm25_searcher: BM25Searcher,
    n_results: int = 5,
) -> List[Dict]:
    """
    Perform hybrid search: combine pre-computed vector results with BM25 search.
    Returns top-n fused results.
    """
    bm25_results = bm25_searcher.search(query, n_results=n_results)
    if not bm25_results:
        return vector_results[:n_results]
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    return fused[:n_results]
