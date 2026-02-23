# AI Voice Agent

Real-time voice conversation with your business documents. Speak naturally, get instant AI answers.

Built with OpenAI Whisper (STT) + GPT-4o (reasoning) + OpenAI TTS (voice output) + FastAPI.

**Author:** Ajay Kumar Reddy Nelavetla | February 2026

## How It Works

1. Paste your business docs
2. Hold the mic button and speak
3. Whisper transcribes instantly
4. GPT-4o answers using your documents
5. TTS speaks back naturally
6. Ready for next question

No Streamlit. No page reloads. Continuous conversation.

## Run Locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
```

## Deploy on Railway

1. Connect this repo
2. Set OPENAI_API_KEY env var
3. Deploy

## Tech Stack

FastAPI, OpenAI Whisper, GPT-4o-mini, OpenAI TTS (Nova), Vanilla HTML/JS
