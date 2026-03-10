"""
AI Voice Agent — Production LLMOps Pipeline
Voice: OpenAI Whisper (STT) + GPT-4o-mini + OpenAI TTS
RAG: Hybrid BM25 + OpenAI Embeddings, Agentic (grade → reformulate → verify)
Observability: LangSmith tracing | Safety: Guardrails | Prompts: Versioned
"""

import io
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from pypdf import PdfReader

from app.agent import agent
from app.voice_handler import transcribe_audio, generate_speech
from app.vector_store import ingest_text, clear_store, get_doc_count

load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "ai-voice-agent")

app = FastAPI(title="AI Voice Agent — Agentic RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DOCUMENT INGESTION ENDPOINTS
# ============================================================

@app.post("/api/upload-text")
async def upload_text_endpoint(text: str = Form(...)):
    """Ingest pasted text into the knowledge base."""
    num_chunks = ingest_text(text.strip(), source="pasted_text")
    agent.rebuild_bm25_index()
    return {"status": "ok", "chunks": num_chunks, "length": len(text.strip())}


@app.post("/api/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """Extract text from PDF and ingest into knowledge base."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        num_chunks = ingest_text(text.strip(), source=file.filename)
        agent.rebuild_bm25_index()
        return {
            "status": "ok",
            "chunks": num_chunks,
            "pages": len(reader.pages),
            "filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/api/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Upload TXT, MD, or PDF file into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    fname = file.filename.lower()
    contents = await file.read()

    if fname.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(contents))
            text = ""
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text += pt + "\n"
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text in PDF")
            num_chunks = ingest_text(text.strip(), source=file.filename)
            agent.rebuild_bm25_index()
            return {
                "status": "ok",
                "chunks": num_chunks,
                "pages": len(reader.pages),
                "filename": file.filename,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif fname.endswith(".txt") or fname.endswith(".md"):
        text = contents.decode("utf-8", errors="ignore").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty file")
        num_chunks = ingest_text(text, source=file.filename)
        agent.rebuild_bm25_index()
        return {"status": "ok", "chunks": num_chunks, "filename": file.filename}

    else:
        raise HTTPException(status_code=400, detail="Supported formats: PDF, TXT, MD")


@app.post("/api/clear")
async def clear_docs():
    """Clear all documents and conversation history."""
    clear_store()
    agent.clear()
    return {"status": "cleared"}


@app.get("/api/stats")
async def get_stats():
    return {
        "doc_count": get_doc_count(),
        "conversation_turns": len(agent.conversation_history) // 2,
    }


# ============================================================
# VOICE PIPELINE ENDPOINT
# ============================================================

@traceable(name="voice_pipeline", run_type="chain")
def run_voice_pipeline(audio_bytes: bytes):
    """Full voice pipeline: STT → Agentic RAG → TTS."""
    start = time.time()

    stt_start = time.time()
    user_text = transcribe_audio(audio_bytes)
    stt_time = round(time.time() - stt_start, 2)

    if not user_text:
        return None, None, None, None, {}

    rag_start = time.time()
    result = agent.query(user_text)
    rag_time = round(time.time() - rag_start, 2)

    answer = result["answer"]

    tts_start = time.time()
    audio_response = generate_speech(answer)
    tts_time = round(time.time() - tts_start, 2)

    metrics = {
        "stt_time": stt_time,
        "rag_time": rag_time,
        "tts_time": tts_time,
        "total_time": round(time.time() - start, 2),
        "confidence": result.get("confidence", 0.5),
        "grounded": result.get("grounded", True),
        "retrieval_attempts": result.get("retrieval_attempts", 1),
        "sources": [s["source"] for s in result.get("sources", [])],
        "blocked": result.get("blocked", False),
        "block_reason": result.get("block_reason", ""),
    }
    return user_text, answer, audio_response, result.get("sources", []), metrics


@app.post("/api/voice")
async def voice_conversation(audio: UploadFile = File(...)):
    """Receive audio, run agentic RAG pipeline, return TTS audio + metadata headers."""
    audio_bytes = await audio.read()
    try:
        user_text, answer, audio_response, sources, metrics = run_voice_pipeline(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not understand audio")

    src_str = ", ".join(s["source"] for s in sources) if sources else ""

    return StreamingResponse(
        io.BytesIO(audio_response),
        media_type="audio/mp3",
        headers={
            "X-User-Text": user_text.replace("\n", " ")[:200],
            "X-Agent-Text": answer.replace("\n", " ")[:500],
            "X-Sources": src_str[:200],
            "X-STT-Time": str(metrics.get("stt_time", 0)),
            "X-RAG-Time": str(metrics.get("rag_time", 0)),
            "X-TTS-Time": str(metrics.get("tts_time", 0)),
            "X-Total-Time": str(metrics.get("total_time", 0)),
            "X-Confidence": str(metrics.get("confidence", 0.5)),
            "X-Grounded": str(metrics.get("grounded", True)),
            "X-Retrieval-Attempts": str(metrics.get("retrieval_attempts", 1)),
            "X-Blocked": str(metrics.get("blocked", False)),
            "Access-Control-Expose-Headers": (
                "X-User-Text, X-Agent-Text, X-Sources, X-STT-Time, X-RAG-Time, "
                "X-TTS-Time, X-Total-Time, X-Confidence, X-Grounded, "
                "X-Retrieval-Attempts, X-Blocked"
            ),
        },
    )


# ============================================================
# TEXT ENDPOINT
# ============================================================

@app.post("/api/text")
async def text_conversation(text: str = Form(...)):
    """Text-only conversation endpoint. Returns enriched JSON with pipeline metadata."""
    result = agent.query(text)
    return JSONResponse(content={
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.5),
        "grounded": result.get("grounded", True),
        "sources": result.get("sources", []),
        "pipeline_trace": result.get("pipeline_trace", []),
        "retrieval_attempts": result.get("retrieval_attempts", 1),
        "latency_seconds": result.get("latency_seconds", 0),
        "blocked": result.get("blocked", False),
    })


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
