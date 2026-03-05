"""
main.py
Sutradhar FastAPI backend.
Run with: uvicorn main:app --reload
"""

import os
import json
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import (
    ask, speech_to_text, text_to_speech,
    LANGUAGE_CODES, STORYTELLERS, SCRIPTURES
)
from auth import authenticate_admin, create_access_token, require_admin
from document_store import add_document, list_documents, delete_document, get_document
from ingest import extract_text, chunk_text, upsert_to_pinecone

load_dotenv()

app = FastAPI(
    title="Sutradhar API",
    description="Multilingual AI storyteller for Indian epics and scriptures",
    version="1.0.0"
)

# ── CORS — allow React Native app and Admin portal to call this API ───────────
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

class LoginRequest(BaseModel):
    username: str
    password: str

class IngestURLRequest(BaseModel):
    url:       str
    scripture: str = "ramayana"
    source:    str = ""
    kanda:     str = ""
    topic:     str = ""


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Sutradhar API"}


# ── Metadata endpoints ────────────────────────────────────────────────────────
@app.get("/scriptures")
def get_scriptures():
    active = [s for s in SCRIPTURES.values() if s.get("active")]
    return {"scriptures": active}


@app.get("/storytellers")
def get_all_storytellers():
    return {"storytellers": list(STORYTELLERS.values())}


@app.get("/storytellers/{scripture}")
def get_storytellers_for_scripture(scripture: str):
    if scripture not in SCRIPTURES:
        raise HTTPException(status_code=404, detail=f"Scripture '{scripture}' not found")
    ids = SCRIPTURES[scripture]["available_storytellers"]
    return {"storytellers": [STORYTELLERS[i] for i in ids if i in STORYTELLERS]}


@app.get("/languages")
def get_languages():
    return {"languages": list(LANGUAGE_CODES.keys())}


# ── Admin auth ────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def admin_login(request: LoginRequest):
    """
    Admin login. Returns JWT token valid for 24 hours.
    Set ADMIN_USERNAME and ADMIN_PASSWORD in environment variables.
    """
    if not authenticate_admin(request.username, request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": request.username, "role": "admin"})
    return {"access_token": token, "token_type": "bearer"}


# ── Core ask endpoint ─────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Ask a question about a scripture."""
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
    """Accept audio file, transcribe via Sarvam STT, then answer."""
    audio_bytes = await audio.read()

    stt_result = speech_to_text(audio_bytes, language=language)
    if stt_result["error"]:
        raise HTTPException(status_code=500, detail=f"STT error: {stt_result['error']}")

    transcript = stt_result["transcript"]
    if not transcript:
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    result = ask(
        query=transcript,
        scripture=scripture,
        storyteller=storyteller,
        language=language
    )

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "transcript":  transcript,
        "answer":      result["answer"],
        "answer_en":   result.get("answer_en"),
        "passages":    result["passages"],
        "scripture":   scripture,
        "storyteller": storyteller,
        "language":    language
    }


# ── TTS endpoint ──────────────────────────────────────────────────────────────
@app.post("/tts")
def text_to_speech_endpoint(request: TTSRequest):
    """Convert text to speech using Sarvam Bulbul v3."""
    import base64
    result = text_to_speech(request.text, language=request.language)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    audio_b64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")
    return {"audio_base64": audio_b64, "format": "wav"}


# ── Document management — admin only ─────────────────────────────────────────

@app.get("/documents")
def get_documents(
    scripture: str = None,
    _admin = Depends(require_admin)
):
    """List all ingested documents. Admin only."""
    docs = list_documents(scripture=scripture)
    return {"documents": docs, "total": len(docs)}


@app.post("/documents/upload")
async def upload_document(
    file:      UploadFile = File(...),
    scripture: str = Form(default="ramayana"),
    source:    str = Form(default=""),
    kanda:     str = Form(default=""),
    topic:     str = Form(default=""),
    _admin = Depends(require_admin)
):
    """
    Upload a PDF, DOCX or TXT file and ingest into Pinecone. Admin only.
    """
    filename  = file.filename or "upload"
    ext       = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .pdf, .docx, .txt"
        )

    if scripture not in SCRIPTURES:
        raise HTTPException(status_code=400, detail=f"Unknown scripture: {scripture}")

    # Save upload to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract + chunk + upsert
        text      = extract_text(tmp_path, is_url=False)
        chunks    = chunk_text(text)
        namespace = SCRIPTURES[scripture]["pinecone_namespace"]
        total     = upsert_to_pinecone(
            chunks=chunks,
            namespace=namespace,
            source=source or filename,
            kanda=kanda,
            topic=topic
        )

        # Track in document store
        doc = add_document(
            source=source or filename,
            scripture=scripture,
            kanda=kanda,
            topic=topic,
            chunk_count=total,
            doc_type="file"
        )

        return {
            "message":     "Document ingested successfully",
            "document_id": doc["id"],
            "filename":    filename,
            "scripture":   scripture,
            "chunk_count": total
        }
    finally:
        os.unlink(tmp_path)


@app.post("/documents/ingest-url")
def ingest_url(
    request: IngestURLRequest,
    _admin = Depends(require_admin)
):
    """
    Ingest content from a web URL into Pinecone. Admin only.
    """
    if request.scripture not in SCRIPTURES:
        raise HTTPException(status_code=400, detail=f"Unknown scripture: {request.scripture}")

    try:
        text      = extract_text(request.url, is_url=True)
        chunks    = chunk_text(text)
        namespace = SCRIPTURES[request.scripture]["pinecone_namespace"]
        total     = upsert_to_pinecone(
            chunks=chunks,
            namespace=namespace,
            source=request.source or request.url,
            kanda=request.kanda,
            topic=request.topic
        )

        doc = add_document(
            source=request.source or request.url,
            scripture=request.scripture,
            kanda=request.kanda,
            topic=request.topic,
            chunk_count=total,
            doc_type="url"
        )

        return {
            "message":     "URL ingested successfully",
            "document_id": doc["id"],
            "url":         request.url,
            "scripture":   request.scripture,
            "chunk_count": total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
def remove_document(
    doc_id: str,
    _admin = Depends(require_admin)
):
    """
    Remove a document record from the document store. Admin only.
    Note: This removes the tracking record. Pinecone vectors are 
    deleted by namespace — use /admin/stats to monitor.
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document(doc_id)
    return {"message": "Document removed", "document_id": doc_id}


# ── Admin stats ───────────────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats(_admin = Depends(require_admin)):
    """
    Returns Pinecone index stats and document counts. Admin only.
    """
    from pinecone import Pinecone
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "sutradhar")

    try:
        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()

        namespaces = {}
        for ns, data in (stats.namespaces or {}).items():
            namespaces[ns] = data.vector_count

        docs = list_documents()

        return {
            "pinecone": {
                "total_vectors": stats.total_vector_count,
                "namespaces":    namespaces
            },
            "documents": {
                "total":      len(docs),
                "by_scripture": {
                    s: len([d for d in docs if d["scripture"] == s])
                    for s in set(d["scripture"] for d in docs)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))