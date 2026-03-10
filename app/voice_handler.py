"""
Voice Handler — OpenAI Whisper STT + TTS Nova.
Extracted from app.py; voice I/O is unchanged from original voice-agent.
"""

import io
import os
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

raw_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
client = wrap_openai(raw_client)


@traceable(name="whisper_transcribe", run_type="llm")
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using OpenAI Whisper. Returns text."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.webm"
    transcript = raw_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
    )
    return transcript.text.strip()


@traceable(name="generate_speech", run_type="tool")
def generate_speech(text: str) -> bytes:
    """Convert text to MP3 audio bytes using OpenAI TTS Nova voice."""
    tts_response = raw_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="mp3",
    )
    return tts_response.content
