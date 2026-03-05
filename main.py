"""
main.py
Sutradhar FastAPI backend.
Run with: uvicorn main:app --reload
"""

import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import (
    ask, speech_to_text, text_to_speech,
    LANGUAGE_CODES, STORYTELLERS, SCRIPTURES
)

load_dotenv()

app = FastAPI(
    title="Sutradhar API",
    description="Multilingual AI storyteller for Indian epics and scriptures",
    version="1.0.0"
)

# ── CORS — allow React Native app to call this API ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:    str
    scripture:   str = "ramayana"
    storyteller: str = "valmiki"
    language:    str = "English"

class AskResponse(BaseModel):
    answer:      str
    answer_en:   str | None = None
    passages:    list[dict]
    scripture:   str
    storyteller: str
    language:    str

class TTSRequest(BaseModel):
    text:     str
    language: str = "English"


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Sutradhar API"}


# ── Metadata endpoints ────────────────────────────────────────────────────────
@app.get("/scriptures")
def get_scriptures():
    """List all available scriptures."""
    active = [s for s in SCRIPTURES.values() if s.get("active")]
    return {"scriptures": active}


@app.get("/storytellers")
def get_all_storytellers():
    """List all storytellers."""
    return {"storytellers": list(STORYTELLERS.values())}


@app.get("/storytellers/{scripture}")
def get_storytellers_for_scripture(scripture: str):
    """List storytellers available for a specific scripture."""
    if scripture not in SCRIPTURES:
        raise HTTPException(status_code=404, detail=f"Scripture '{scripture}' not found")
    ids = SCRIPTURES[scripture]["available_storytellers"]
    return {"storytellers": [STORYTELLERS[i] for i in ids if i in STORYTELLERS]}


@app.get("/languages")
def get_languages():
    """List all supported languages."""
    return {"languages": list(LANGUAGE_CODES.keys())}


# ── Core ask endpoint ─────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Ask a question about a scripture.
    Returns answer in the requested language with source passages.
    """
    if request.scripture not in SCRIPTURES:
        raise HTTPException(status_code=400, detail=f"Unknown scripture: {request.scripture}")

    if request.storyteller not in STORYTELLERS:
        raise HTTPException(status_code=400, detail=f"Unknown storyteller: {request.storyteller}")

    if request.language not in LANGUAGE_CODES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

    result = ask(
        query=request.question,
        scripture=request.scripture,
        storyteller=request.storyteller,
        language=request.language
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return AskResponse(
        answer=result["answer"],
        answer_en=result.get("answer_en"),
        passages=result["passages"],
        scripture=result["scripture"],
        storyteller=result["storyteller"],
        language=result["language"]
    )


# ── Voice input endpoint ──────────────────────────────────────────────────────
@app.post("/ask/voice")
async def ask_by_voice(
    audio:       UploadFile = File(...),
    scripture:   str = "ramayana",
    storyteller: str = "valmiki",
    language:    str = "English"
):
    """
    Accept audio file, transcribe via Sarvam STT, then answer.
    Returns both the transcript and the answer.
    """
    audio_bytes = await audio.read()

    # Step 1 — Transcribe
    stt_result = speech_to_text(audio_bytes, language=language)
    if stt_result["error"]:
        raise HTTPException(status_code=500, detail=f"STT error: {stt_result['error']}")

    transcript = stt_result["transcript"]
    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    # Step 2 — Answer
    result = ask(
        query=transcript,
        scripture=scripture,
        storyteller=storyteller,
        language=language
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "transcript": transcript,
        "answer":     result["answer"],
        "answer_en":  result.get("answer_en"),
        "passages":   result["passages"],
        "scripture":  scripture,
        "storyteller": storyteller,
        "language":   language
    }


# ── TTS endpoint ──────────────────────────────────────────────────────────────
@app.post("/tts")
def text_to_speech_endpoint(request: TTSRequest):
    """
    Convert text to speech using Sarvam Bulbul v3.
    Returns base64-encoded WAV audio.
    """
    import base64
    result = text_to_speech(request.text, language=request.language)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    audio_b64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")
    return {"audio_base64": audio_b64, "format": "wav"}