"""
Vector Store — Async version with batch embeddings and EmbeddingCache.
Chunking + OpenAI Embeddings + In-memory cosine similarity search.
Lightweight: no ChromaDB, no PyTorch. Runs on Railway free tier.

Phase 11: ingest_text now batches ALL chunks in a single embeddings API call.
Phase 4:  get_embedding uses EmbeddingCache to skip repeated API calls.
"""

import os
import numpy as np
from openai import AsyncOpenAI

from app.cache import embedding_cache

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# In-memory store: list of {"text": str, "source": str, "embedding": np.array}
chunks_store: list = []

CHUNK_SIZE = 500   # words
CHUNK_OVERLAP = 50  # words
EMBEDDING_BATCH_SIZE = 100  # OpenAI max is 2048 inputs, 100 is safe


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


async def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for text, with EmbeddingCache to skip repeat calls."""
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    embedding_cache.set(text, emb)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def ingest_text(text: str, source: str = "pasted_text") -> int:
    """
    Chunk text, batch-embed with OpenAI (single API call per 100 chunks),
    and store in memory. Returns chunk count.

    Phase 11: sends all chunk texts in one embeddings.create() call
    instead of one call per chunk — dramatically faster for large PDFs.
    """
    global chunks_store
    new_chunks = chunk_text(text, source)
    if not new_chunks:
        return 0

    texts = [c["text"] for c in new_chunks]
    all_embeddings: list = []

    # Batch into groups of EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        # response.data is sorted by index
        for emb_obj in response.data:
            all_embeddings.append(np.array(emb_obj.embedding, dtype=np.float32))

    for chunk, emb in zip(new_chunks, all_embeddings):
        chunks_store.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": emb,
        })

    return len(new_chunks)


async def retrieve(query: str, k: int = 5) -> list:
    """
    Find top-k most similar chunks to query via cosine similarity.
    Returns list of dicts: {"text", "source", "relevance_score"}
    """
    if not chunks_store:
        return []
    query_emb = await get_embedding(query)
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


async def get_context_for_query(query: str, k: int = 5) -> tuple:
    """Get formatted context string + sources list for direct LLM use."""
    results = await retrieve(query, k)
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
