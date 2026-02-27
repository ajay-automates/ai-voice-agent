<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,30,35&height=170&section=header&text=AI%20Voice%20Agent&fontSize=52&fontAlignY=35&animation=twinkling&fontColor=ffffff&desc=Real-Time%20Voice%20Conversations%20with%20Your%20Documents&descAlignY=55&descSize=18" width="100%" />

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](.)
[![OpenAI Whisper](https://img.shields.io/badge/Whisper-STT-412991?style=for-the-badge&logo=openai&logoColor=white)](.)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-Reasoning-412991?style=for-the-badge&logo=openai&logoColor=white)](.)
[![TTS](https://img.shields.io/badge/OpenAI-TTS%20Nova-412991?style=for-the-badge&logo=openai&logoColor=white)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Speak naturally. Get instant AI answers from your business documents. No page reloads.**

</div>

---

## Why This Exists

Most document Q&A tools are text-based — copy, paste, wait, read. This project lets you **talk** to your documents like a colleague. Hold the mic, ask your question, get a spoken answer back in seconds.

No Streamlit. No page reloads. Continuous, real-time voice conversation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser (Vanilla JS)                      │
│                                                               │
│   ┌──────────────┐    ┌──────────┐    ┌───────────────────┐  │
│   │  Mic Button   │    │  Audio   │    │  Document Paste   │  │
│   │  (hold-to-    │    │  Player  │    │  (business docs)  │  │
│   │   record)     │    │  (TTS)   │    │                   │  │
│   └──────┬───────┘    └────▲─────┘    └───────────────────┘  │
│          │                  │                                  │
└──────────┼──────────────────┼──────────────────────────────────┘
           │ audio blob        │ audio/mpeg
           ▼                   │
┌──────────────────────────────┼──────────────────────────────┐
│              FastAPI Server   │                               │
│                               │                               │
│   ┌────────────┐    ┌───────┴────────┐    ┌──────────────┐  │
│   │  Whisper   │    │    OpenAI      │    │   OpenAI     │  │
│   │  (STT)     │──→ │    GPT-4o     │──→ │   TTS Nova   │  │
│   │  transcribe│    │  + doc context │    │   synthesize  │  │
│   └────────────┘    └────────────────┘    └──────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## How It Works

| Step | What Happens | Technology |
|------|-------------|------------|
| **1** | Paste your business documents into the text area | Browser |
| **2** | Hold the mic button and speak your question | MediaRecorder API |
| **3** | Audio sent to server, transcribed in real-time | OpenAI Whisper |
| **4** | Question + document context sent to LLM | GPT-4o-mini |
| **5** | Response synthesized to natural speech | OpenAI TTS (Nova voice) |
| **6** | Audio streamed back, plays automatically | HTML5 Audio |
| **7** | Ready for next question — continuous conversation | WebSocket-like loop |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI over Flask** | Async support for concurrent audio streaming |
| **Vanilla JS over React** | Zero build step, instant load, minimal complexity for audio handling |
| **Whisper over browser STT** | Higher accuracy, supports accents, no browser compatibility issues |
| **GPT-4o-mini over GPT-4o** | 80% cost reduction with minimal quality loss for document Q&A |
| **TTS Nova voice** | Most natural-sounding voice in OpenAI's lineup |
| **No database** | Stateless — documents live in session memory, no PII stored |

---

## Quick Start

```bash
git clone https://github.com/ajay-automates/ai-voice-agent.git
cd ai-voice-agent
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
# Open http://localhost:7860
```

### Deploy on Railway

1. Connect this GitHub repo to Railway
2. Set `OPENAI_API_KEY` as environment variable
3. Deploy — Railway auto-detects Python

---

## Project Structure

```
ai-voice-agent/
├── app.py                # FastAPI server — Whisper + GPT-4o + TTS pipeline
├── templates/
│   └── index.html        # Single-page UI with mic button + audio player
├── requirements.txt      # FastAPI, OpenAI, uvicorn, python-multipart
└── README.md
```

---

## Cost Analysis

| Component | Cost |
|-----------|------|
| Whisper STT | ~$0.006/min of audio |
| GPT-4o-mini | ~$0.0015/query |
| TTS Nova | ~$0.015/1K chars |
| **Per conversation (5 questions)** | **~$0.10** |
| **Monthly (50 conversations/day)** | **~$150** |

---

## Tech Stack

`FastAPI` `OpenAI Whisper` `GPT-4o-mini` `OpenAI TTS` `Python` `Vanilla JavaScript` `HTML5 Audio API` `Railway`

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AI Support Agent](https://github.com/ajay-automates/ai-support-agent) | RAG chatbot with LangSmith observability |
| [AI Image Classifier API](https://github.com/ajay-automates/ai-image-classifier-api) | Self-hosted CLIP inference — $0/request |
| [EazyApply](https://github.com/ajay-automates/eazyapply) | Chrome extension for auto-filling job applications |

---

<div align="center">

**Built by [Ajay Kumar Reddy Nelavetla](https://github.com/ajay-automates)** · February 2026

*Talk to your documents. Get answers instantly.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,30,35&height=100&section=footer" width="100%" />

</div>
