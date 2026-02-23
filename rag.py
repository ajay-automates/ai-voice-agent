"""
RAG Engine (Lightweight) — Chunking + OpenAI Embeddings + In-memory vector search.
No ChromaDB, no PyTorch, no sentence-transformers. Runs on Railway free tier.
"""

import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# In-memory vector store
chunks_store = []  # list of {"text": str, "source": str, "embedding": np.array}

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def chunk_text(text, source="pasted_text"):
    """Split text into overlapping chunks."""
    sentences = text.replace("\n\n", "\n").split("\n")
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) > CHUNK_SIZE:
            if current:
                chunks.append({"text": current.strip(), "source": source})
            current = s
        else:
            current = current + "\n" + s if current else s
    if current.strip():
        chunks.append({"text": current.strip(), "source": source})
    return chunks


def get_embedding(text):
    """Get OpenAI embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ingest_text(text, source="pasted_text"):
    """Chunk text, embed with OpenAI, store in memory."""
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


def retrieve(query, k=4):
    """Find top-k most similar chunks to query."""
    if not chunks_store:
        return []
    query_emb = get_embedding(query)
    scored = []
    for chunk in chunks_store:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        scored.append({"text": chunk["text"], "source": chunk["source"], "score": float(sim)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def get_context_for_query(query, k=4):
    """Get formatted context string with sources."""
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


def clear_store():
    global chunks_store
    chunks_store = []


def get_doc_count():
    return len(chunks_store)
