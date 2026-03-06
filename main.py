"""
main.py — Sutradhar FastAPI backend
Run with: uvicorn main:app --reload
"""

import os
import json
import tempfile
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import ask, speech_to_text, text_to_speech, LANGUAGE_CODES, STORYTELLERS, SCRIPTURES
from auth import authenticate_admin, create_access_token, require_admin
from document_store import (
    add_document, list_documents, delete_document, get_document,
    update_document, create_job, update_job, get_job
)
from ingest import extract_text, chunk_text, upsert_to_pinecone

load_dotenv()

app = FastAPI(
    title="Sutradhar API",
    description="Multilingual AI storyteller for Indian epics and scriptures",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
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

class UpdateDocRequest(BaseModel):
    source:    str | None = None
    kanda:     str | None = None
    topic:     str | None = None
    scripture: str | None = None


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Sutradhar API"}


# ── Metadata ──────────────────────────────────────────────────────────────────
@app.get("/scriptures")
def get_scriptures():
    return {"scriptures": [s for s in SCRIPTURES.values() if s.get("active")]}

@app.get("/storytellers")
def get_all_storytellers():
    return {"storytellers": list(STORYTELLERS.values())}

@app.get("/storytellers/{scripture}")
def get_storytellers_for_scripture(scripture: str):
    if scripture not in SCRIPTURES:
        raise HTTPException(404, f"Scripture '{scripture}' not found")
    ids = SCRIPTURES[scripture]["available_storytellers"]
    return {"storytellers": [STORYTELLERS[i] for i in ids if i in STORYTELLERS]}

@app.get("/languages")
def get_languages():
    return {"languages": list(LANGUAGE_CODES.keys())}


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def admin_login(request: LoginRequest):
    if not authenticate_admin(request.username, request.password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": request.username, "role": "admin"})
    return {"access_token": token, "token_type": "bearer"}


# ── Ask ───────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if request.scripture   not in SCRIPTURES:   raise HTTPException(400, f"Unknown scripture: {request.scripture}")
    if request.storyteller not in STORYTELLERS: raise HTTPException(400, f"Unknown storyteller: {request.storyteller}")
    if request.language    not in LANGUAGE_CODES: raise HTTPException(400, f"Unsupported language: {request.language}")
    result = ask(query=request.question, scripture=request.scripture, storyteller=request.storyteller, language=request.language)
    if result["error"]: raise HTTPException(500, result["error"])
    return AskResponse(answer=result["answer"], answer_en=result.get("answer_en"), passages=result["passages"], scripture=result["scripture"], storyteller=result["storyteller"], language=result["language"])


@app.post("/ask/voice")
async def ask_by_voice(audio: UploadFile = File(...), scripture: str = "ramayana", storyteller: str = "valmiki", language: str = "English"):
    audio_bytes = await audio.read()
    stt = speech_to_text(audio_bytes, language=language)
    if stt["error"]: raise HTTPException(500, f"STT error: {stt['error']}")
    if not stt["transcript"]: raise HTTPException(400, "Could not transcribe audio")
    result = ask(query=stt["transcript"], scripture=scripture, storyteller=storyteller, language=language)
    if result["error"]: raise HTTPException(500, result["error"])
    return {"transcript": stt["transcript"], "answer": result["answer"], "answer_en": result.get("answer_en"), "passages": result["passages"], "scripture": scripture, "storyteller": storyteller, "language": language}


@app.post("/tts")
def tts_endpoint(request: TTSRequest):
    import base64
    result = text_to_speech(request.text, language=request.language)
    if result["error"]: raise HTTPException(500, result["error"])
    return {"audio_base64": base64.b64encode(result["audio_bytes"]).decode(), "format": "wav"}


# ── Documents ─────────────────────────────────────────────────────────────────
@app.get("/documents")
def get_documents(scripture: str = None, search: str = None, page: int = 1, page_size: int = 20, _admin=Depends(require_admin)):
    docs = list_documents(scripture=scripture)
    # Search filter
    if search:
        q = search.lower()
        docs = [d for d in docs if q in d.get("source","").lower() or q in d.get("kanda","").lower() or q in d.get("topic","").lower()]
    # Pagination
    total = len(docs)
    start = (page - 1) * page_size
    end   = start + page_size
    return {
        "documents": docs[start:end],
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size
    }


def _run_ingestion(job_id, tmp_path, ext, scripture, source, kanda, topic, filename):
    """Background ingestion with progress updates."""
    try:
        update_job(job_id, status="processing", progress=10, message="Extracting text...")
        text   = extract_text(tmp_path, is_url=False)

        update_job(job_id, progress=30, message="Chunking document...")
        chunks = chunk_text(text)

        update_job(job_id, progress=50, message=f"Upserting {len(chunks)} chunks to Pinecone...")
        namespace = SCRIPTURES[scripture]["pinecone_namespace"]
        total  = upsert_to_pinecone(chunks=chunks, namespace=namespace, source=source, kanda=kanda, topic=topic)

        update_job(job_id, progress=90, message="Saving document record...")
        doc = add_document(source=source, scripture=scripture, kanda=kanda, topic=topic, chunk_count=total, doc_type="file")

        update_job(job_id, status="done", progress=100, message=f"Done — {total} chunks ingested", chunk_count=total, doc_id=doc["id"])
    except Exception as e:
        update_job(job_id, status="error", message=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass


@app.post("/documents/upload")
async def upload_document(
    file:      UploadFile = File(...),
    scripture: str = Form(default="ramayana"),
    source:    str = Form(default=""),
    kanda:     str = Form(default=""),
    topic:     str = Form(default=""),
    _admin=Depends(require_admin)
):
    filename = file.filename or "upload"
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if scripture not in SCRIPTURES:
        raise HTTPException(400, f"Unknown scripture: {scripture}")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Create job and start background thread
    job = create_job(filename=filename, scripture=scripture)
    thread = threading.Thread(
        target=_run_ingestion,
        args=(job["id"], tmp_path, ext, scripture, source or filename, kanda, topic, filename),
        daemon=True
    )
    thread.start()

    return {"job_id": job["id"], "message": "Ingestion started", "filename": filename}


@app.get("/documents/jobs/{job_id}")
def get_job_status(job_id: str, _admin=Depends(require_admin)):
    """Poll this endpoint to track ingestion progress."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/documents/ingest-url")
def ingest_url(request: IngestURLRequest, _admin=Depends(require_admin)):
    if request.scripture not in SCRIPTURES:
        raise HTTPException(400, f"Unknown scripture: {request.scripture}")
    try:
        text      = extract_text(request.url, is_url=True)
        chunks    = chunk_text(text)
        namespace = SCRIPTURES[request.scripture]["pinecone_namespace"]
        total     = upsert_to_pinecone(chunks=chunks, namespace=namespace, source=request.source or request.url, kanda=request.kanda, topic=request.topic)
        doc       = add_document(source=request.source or request.url, scripture=request.scripture, kanda=request.kanda, topic=request.topic, chunk_count=total, doc_type="url")
        return {"message": "URL ingested successfully", "document_id": doc["id"], "url": request.url, "scripture": request.scripture, "chunk_count": total}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.patch("/documents/{doc_id}")
def edit_document(doc_id: str, request: UpdateDocRequest, _admin=Depends(require_admin)):
    """Edit document metadata — source, kanda, topic, scripture."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    updated = update_document(doc_id, updates)
    return {"message": "Document updated", "document": updated}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str, _admin=Depends(require_admin)):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    delete_document(doc_id)
    return {"message": "Document removed", "document_id": doc_id}


# ── Admin stats ───────────────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats(_admin=Depends(require_admin)):
    from pinecone import Pinecone
    try:
        pc    = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(os.environ.get("PINECONE_INDEX", "sutradhar"))
        stats = index.describe_index_stats()
        namespaces = {ns: data.vector_count for ns, data in (stats.namespaces or {}).items()}
        docs  = list_documents()
        return {
            "pinecone":  {"total_vectors": stats.total_vector_count, "namespaces": namespaces},
            "documents": {"total": len(docs), "by_scripture": {s: len([d for d in docs if d["scripture"] == s]) for s in set(d["scripture"] for d in docs)}}
        }
    except Exception as e:
        raise HTTPException(500, str(e))