"""
Voice Handler — Async OpenAI Whisper STT + TTS Nova.

Phase 2:  Switched to AsyncOpenAI — non-blocking audio API calls.
Phase 12: generate_speech_stream() yields MP3 chunks as OpenAI generates them,
          enabling /api/voice/stream to send audio before full MP3 is ready.
"""

import io
import os
from openai import AsyncOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

raw_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = wrap_openai(raw_client)


@traceable(name="whisper_transcribe", run_type="llm")
async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using OpenAI Whisper. Returns text or empty string."""
    if len(audio_bytes) < 1000:
        return ""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.webm"
        transcript = await raw_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
        )
        return transcript.text.strip()
    except Exception:
        return ""


@traceable(name="generate_speech", run_type="tool")
async def generate_speech(text: str) -> bytes:
    """Convert text to MP3 audio bytes using OpenAI TTS Nova voice."""
    tts_response = await raw_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="mp3",
    )
    return tts_response.content


async def generate_speech_stream(text: str):
    """
    Stream MP3 audio chunks from OpenAI TTS as they are generated (Phase 12).
    Client receives first bytes ~200ms after TTS starts instead of waiting
    for the full MP3 to be generated (~1-2s).
    """
    async with raw_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="mp3",
    ) as response:
        async for chunk in response.iter_bytes(4096):
            yield chunk
