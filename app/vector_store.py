"""
Vector Store — Chunking + OpenAI Embeddings + In-memory cosine similarity search.
Lightweight: no ChromaDB, no PyTorch. Runs on Railway free tier.
"""

import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# In-memory store: list of {"text": str, "source": str, "embedding": np.array}
chunks_store: list = []

CHUNK_SIZE = 500   # words (upgraded from 400 chars)
CHUNK_OVERLAP = 50  # words


def chunk_text(text: str, source: str = "pasted_text") -> list:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text_str = " ".join(chunk_words).strip()
        if chunk_text_str:
            chunks.append({"text": chunk_text_str, "source": source})
        i += CHUNK_SIZE - CHUNK_OVERLAP
        if i >= len(words):
            break
    return chunks


def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def ingest_text(text: str, source: str = "pasted_text") -> int:
    """Chunk text, embed with OpenAI, store in memory. Returns chunk count."""
    global chunks_store
    new_chunks = chunk_text(text, source)
    for chunk in new_chunks:
        emb = get_embedding(chunk["text"])
        chunks_store.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": emb,
        })
    return len(new_chunks)


def retrieve(query: str, k: int = 5) -> list:
    """
    Find top-k most similar chunks to query.
    Returns list of dicts: {"text", "source", "relevance_score"}
    """
    if not chunks_store:
        return []
    query_emb = get_embedding(query)
    scored = []
    for chunk in chunks_store:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        scored.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "relevance_score": round(sim, 4),
        })
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:k]


def get_context_for_query(query: str, k: int = 5) -> tuple:
    """Get formatted context string + sources list for direct LLM use."""
    results = retrieve(query, k)
    if not results:
        return "No relevant documents found.", []
    parts = []
    sources = []
    for r in results:
        parts.append(f"[Source: {r['source']}]\n{r['text']}")
        if r["source"] not in sources:
            sources.append(r["source"])
    return "\n\n---\n\n".join(parts), sources


def get_all_chunks() -> list:
    """Return all stored chunks (for BM25 index building)."""
    return [{"text": c["text"], "source": c["source"]} for c in chunks_store]


def clear_store() -> None:
    global chunks_store
    chunks_store = []


def get_doc_count() -> int:
    return len(chunks_store)
