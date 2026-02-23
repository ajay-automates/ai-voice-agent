"""
AI Voice Agent - Real-time voice conversation with documents.
OpenAI Whisper (STT) + GPT-4o-mini (reasoning) + OpenAI TTS (voice output).
"""

import os
import io
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

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


@app.post("/api/voice")
async def voice_conversation(audio: UploadFile = File(...)):
    global conversation_history
    start_time = time.time()

    audio_bytes = await audio.read()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.webm"

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language="en",
        )
        user_text = transcript.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not understand audio")

    stt_time = time.time() - start_time

    system_prompt = f"""You are a helpful, friendly customer support agent. Answer questions using the provided business documents. Be concise - keep answers to 2-3 sentences since this will be spoken aloud. If the answer is not in the documents, say so briefly.

Business Documents:
---
{document_context[:8000]}
---"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in conversation_history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_text})

    try:
        llm_start = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.3,
        )
        answer = completion.choices[0].message.content.strip()
        llm_time = time.time() - llm_start
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": answer})

    try:
        tts_start = time.time()
        tts_response = client.audio.speech.create(
            model="tts-1", voice="nova", input=answer, response_format="mp3",
        )
        audio_response = tts_response.content
        tts_time = time.time() - tts_start
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    total_time = time.time() - start_time

    return StreamingResponse(
        io.BytesIO(audio_response),
        media_type="audio/mp3",
        headers={
            "X-User-Text": user_text.replace('\n', ' ')[:200],
            "X-Agent-Text": answer.replace('\n', ' ')[:500],
            "X-STT-Time": str(round(stt_time, 2)),
            "X-LLM-Time": str(round(llm_time, 2)),
            "X-TTS-Time": str(round(tts_time, 2)),
            "X-Total-Time": str(round(total_time, 2)),
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
