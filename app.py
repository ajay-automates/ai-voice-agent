"""
AI Voice Agent — Full LLMOps Pipeline (Lightweight)
Voice: OpenAI Whisper (STT) + GPT-4o-mini + OpenAI TTS
RAG: OpenAI Embeddings + Numpy cosine similarity (no heavy deps)
Observability: LangSmith tracing
Safety: Guardrails for prompt injection
Prompts: Versioned templates
"""

import os
import io
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

from rag import ingest_text, get_context_for_query, clear_store, get_doc_count
from guardrails import check_guardrails

load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "ai-voice-agent")

raw_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = wrap_openai(raw_client)

conversation_history = []

app = FastAPI(title="AI Voice Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_prompt(version="v2_voice"):
    prompt_path = Path(__file__).parent / "prompts" / f"{version}.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are a helpful customer support agent. Answer using only the provided context."


@app.post("/api/upload-text")
async def upload_text_endpoint(text: str = Form(...)):
    num_chunks = ingest_text(text.strip(), source="pasted_text")
    return {"status": "ok", "chunks": num_chunks, "length": len(text.strip())}


@app.post("/api/clear")
async def clear_docs():
    global conversation_history
    clear_store()
    conversation_history = []
    return {"status": "cleared"}


@app.get("/api/stats")
async def get_stats():
    return {"doc_count": get_doc_count(), "conversation_turns": len(conversation_history) // 2}


@traceable(name="whisper_transcribe", run_type="llm")
def transcribe_audio(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.webm"
    transcript = raw_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="en",
    )
    return transcript.text.strip()


@traceable(name="rag_retrieve", run_type="retriever")
def retrieve_context(query):
    context, sources = get_context_for_query(query, k=4)
    return context, sources


@traceable(name="generate_answer", run_type="chain")
def generate_answer(user_text, context, sources, history):
    system_prompt = load_prompt("v2_voice")
    full_prompt = f"""{system_prompt}

Retrieved Context:
---
{context}
---

Sources: {', '.join(sources) if sources else 'None'}"""

    messages = [{"role": "system", "content": full_prompt}]
    for h in history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


@traceable(name="generate_speech", run_type="tool")
def generate_speech(text):
    tts_response = raw_client.audio.speech.create(
        model="tts-1", voice="nova", input=text, response_format="mp3",
    )
    return tts_response.content


@traceable(name="voice_pipeline", run_type="chain")
def voice_pipeline(audio_bytes):
    global conversation_history
    start = time.time()

    stt_start = time.time()
    user_text = transcribe_audio(audio_bytes)
    stt_time = time.time() - stt_start
    if not user_text:
        return None, None, None, None, {}

    guard = check_guardrails(user_text)
    if not guard["allowed"]:
        blocked_answer = f"I can't process that request. {guard['reason']}."
        tts_bytes = generate_speech(blocked_answer)
        return user_text, blocked_answer, tts_bytes, [], {
            "stt_time": round(stt_time, 2), "blocked": True,
            "block_reason": guard["reason"], "total_time": round(time.time() - start, 2),
        }

    rag_start = time.time()
    context, sources = retrieve_context(user_text)
    rag_time = time.time() - rag_start

    llm_start = time.time()
    answer = generate_answer(user_text, context, sources, conversation_history)
    llm_time = time.time() - llm_start

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": answer})

    tts_start = time.time()
    audio_response = generate_speech(answer)
    tts_time = time.time() - tts_start

    metrics = {
        "stt_time": round(stt_time, 2), "rag_time": round(rag_time, 2),
        "llm_time": round(llm_time, 2), "tts_time": round(tts_time, 2),
        "total_time": round(time.time() - start, 2),
        "chunks_retrieved": len(sources), "sources": sources, "blocked": False,
    }
    return user_text, answer, audio_response, sources, metrics


@app.post("/api/voice")
async def voice_conversation(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    try:
        user_text, answer, audio_response, sources, metrics = voice_pipeline(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not user_text:
        raise HTTPException(status_code=400, detail="Could not understand audio")
    src_str = ", ".join(sources) if sources else ""
    return StreamingResponse(
        io.BytesIO(audio_response), media_type="audio/mp3",
        headers={
            "X-User-Text": user_text.replace('\n', ' ')[:200],
            "X-Agent-Text": answer.replace('\n', ' ')[:500],
            "X-Sources": src_str[:200],
            "X-STT-Time": str(metrics.get("stt_time", 0)),
            "X-RAG-Time": str(metrics.get("rag_time", 0)),
            "X-LLM-Time": str(metrics.get("llm_time", 0)),
            "X-TTS-Time": str(metrics.get("tts_time", 0)),
            "X-Total-Time": str(metrics.get("total_time", 0)),
            "X-Blocked": str(metrics.get("blocked", False)),
            "Access-Control-Expose-Headers": "X-User-Text, X-Agent-Text, X-Sources, X-STT-Time, X-RAG-Time, X-LLM-Time, X-TTS-Time, X-Total-Time, X-Blocked",
        },
    )


@app.post("/api/text")
async def text_conversation(text: str = Form(...)):
    global conversation_history
    guard = check_guardrails(text)
    if not guard["allowed"]:
        return {"answer": f"I can't process that request. {guard['reason']}.", "blocked": True}
    context, sources = retrieve_context(text)
    system_prompt = load_prompt("v2_voice")
    full_prompt = f"""{system_prompt}\n\nContext:\n---\n{context}\n---"""
    messages = [{"role": "system", "content": full_prompt}]
    for h in conversation_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": text})
    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.3,
    )
    answer = completion.choices[0].message.content.strip()
    conversation_history.append({"role": "user", "content": text})
    conversation_history.append({"role": "assistant", "content": answer})
    return {"answer": answer, "sources": sources, "blocked": False}


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
