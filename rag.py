"""
RAG Engine — Document ingestion, chunking, embedding, retrieval.
Uses LangChain + ChromaDB + HuggingFace embeddings.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def get_vectorstore():
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def ingest_text(text, source="pasted_text"):
    """Chunk text and store embeddings in ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    return len(docs)


def retrieve(query, k=4):
    """Retrieve top-k relevant chunks for a query."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        })
    return results


def get_context_for_query(query, k=4):
    """Get formatted context string with source citations."""
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant documents found.", []

    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks):
        src = chunk["source"]
        context_parts.append(f"[Source: {src}]\n{chunk['content']}")
        if src not in sources:
            sources.append(src)

    return "\n\n---\n\n".join(context_parts), sources


def clear_vectorstore():
    """Clear all documents."""
    vectorstore = get_vectorstore()
    vectorstore.delete_collection()


def get_doc_count():
    """Get number of indexed chunks."""
    try:
        vectorstore = get_vectorstore()
        return vectorstore._collection.count()
    except Exception:
        return 0
