"""
AI Voice Agent - Real-time voice conversation with documents.
OpenAI Whisper (STT) + GPT-4o-mini (reasoning) + OpenAI TTS (voice output).
Full observability via LangSmith tracing.
"""

import os
import io
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

load_dotenv()

# LangSmith config
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "ai-voice-agent")

# Wrap OpenAI client with LangSmith tracing
raw_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = wrap_openai(raw_client)

document_context = "No documents uploaded yet. Please upload business documents to enable AI-powered answers."
conversation_history = []

app = FastAPI(title="AI Voice Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/upload-text")
async def upload_text(text: str = Form(...)):
    global document_context
    document_context = text.strip()
    return {"status": "ok", "length": len(document_context)}


@app.post("/api/clear")
async def clear_docs():
    global document_context, conversation_history
    document_context = "No documents uploaded yet."
    conversation_history = []
    return {"status": "cleared"}


@traceable(name="whisper_transcribe", run_type="llm")
def transcribe_audio(audio_bytes):
    """Transcribe audio with Whisper - traced by LangSmith."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.webm"
    transcript = raw_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="en",
    )
    return transcript.text.strip()


@traceable(name="generate_answer", run_type="chain")
def generate_answer(user_text, doc_context, history):
    """Generate answer with GPT-4o-mini - traced by LangSmith."""
    system_prompt = f"""You are a helpful, friendly customer support agent. Answer questions using the provided business documents. Be concise - keep answers to 2-3 sentences since this will be spoken aloud. If the answer is not in the documents, say so briefly.

Business Documents:
---
{doc_context[:8000]}
---"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


@traceable(name="generate_speech", run_type="tool")
def generate_speech(text):
    """Generate TTS audio - traced by LangSmith."""
    tts_response = raw_client.audio.speech.create(
        model="tts-1", voice="nova", input=text, response_format="mp3",
    )
    return tts_response.content


@traceable(name="voice_pipeline", run_type="chain")
def voice_pipeline(audio_bytes):
    """Full voice pipeline: STT -> LLM -> TTS. All traced."""
    global conversation_history
    start_time = time.time()

    # Step 1: Transcribe
    stt_start = time.time()
    user_text = transcribe_audio(audio_bytes)
    stt_time = time.time() - stt_start

    if not user_text:
        return None, None, None, {}

    # Step 2: Generate answer
    llm_start = time.time()
    answer = generate_answer(user_text, document_context, conversation_history)
    llm_time = time.time() - llm_start

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": answer})

    # Step 3: Generate speech
    tts_start = time.time()
    audio_response = generate_speech(answer)
    tts_time = time.time() - tts_start

    total_time = time.time() - start_time

    metrics = {
        "stt_time": round(stt_time, 2),
        "llm_time": round(llm_time, 2),
        "tts_time": round(tts_time, 2),
        "total_time": round(total_time, 2),
    }

    return user_text, answer, audio_response, metrics


@app.post("/api/voice")
async def voice_conversation(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    try:
        user_text, answer, audio_response, metrics = voice_pipeline(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not understand audio")

    return StreamingResponse(
        io.BytesIO(audio_response),
        media_type="audio/mp3",
        headers={
            "X-User-Text": user_text.replace('\n', ' ')[:200],
            "X-Agent-Text": answer.replace('\n', ' ')[:500],
            "X-STT-Time": str(metrics["stt_time"]),
            "X-LLM-Time": str(metrics["llm_time"]),
            "X-TTS-Time": str(metrics["tts_time"]),
            "X-Total-Time": str(metrics["total_time"]),
            "Access-Control-Expose-Headers": "X-User-Text, X-Agent-Text, X-STT-Time, X-LLM-Time, X-TTS-Time, X-Total-Time",
        },
    )


@app.post("/api/text")
async def text_conversation(text: str = Form(...)):
    global conversation_history
    system_prompt = f"""You are a helpful customer support agent. Answer using the business documents. Be concise.

Documents:
---
{document_context[:8000]}
---"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in conversation_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": text})

    completion = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.3,
    )
    answer = completion.choices[0].message.content.strip()
    conversation_history.append({"role": "user", "content": text})
    conversation_history.append({"role": "assistant", "content": answer})
    return {"answer": answer}


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    print("AI Voice Agent starting on http://localhost:8000")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
