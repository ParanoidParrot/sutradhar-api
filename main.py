"""
main.py — Sutradhar FastAPI backend
"""

import os
import json
import tempfile
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from rag import ask, speech_to_text, text_to_speech, LANGUAGE_CODES, STORYTELLERS, SCRIPTURES
from auth import authenticate_admin, create_access_token, require_admin
from document_store import (
    add_document, list_documents, delete_document, get_document,
    update_document, create_job, update_job, get_job
)
from activity_store import log as activity_log, get_log
from admin_users import list_users, create_user, update_user, delete_user, get_user
from config_store import (
    list_scriptures, get_scripture, create_scripture, update_scripture,
    list_storytellers, get_storyteller, create_storyteller, update_storyteller, delete_storyteller
)
from ingest import extract_text, chunk_text, upsert_to_pinecone

load_dotenv()

app = FastAPI(title="Sutradhar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str; scripture: str = "ramayana"; storyteller: str = "valmiki"; language: str = "English"

class TTSRequest(BaseModel):
    text: str; language: str = "English"

class LoginRequest(BaseModel):
    username: str; password: str

class IngestURLRequest(BaseModel):
    url: str; scripture: str = "ramayana"; source: str = ""; kanda: str = ""; topic: str = ""

class UpdateDocRequest(BaseModel):
    source: Optional[str] = None; kanda: Optional[str] = None; topic: Optional[str] = None; scripture: Optional[str] = None

class CreateUserRequest(BaseModel):
    username: str; password: str; display_name: str = ""

class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = None; password: Optional[str] = None; active: Optional[bool] = None

class CreateScriptureRequest(BaseModel):
    id: str; name: str; description: str = ""; pinecone_namespace: str; default_storyteller: str; available_storytellers: list[str] = []

class UpdateScriptureRequest(BaseModel):
    name: Optional[str] = None; description: Optional[str] = None; pinecone_namespace: Optional[str] = None
    default_storyteller: Optional[str] = None; available_storytellers: Optional[list[str]] = None; active: Optional[bool] = None

class CreateStorytellerRequest(BaseModel):
    id: str; name: str; scripture: str; system_prompt: str; greeting: str; tone: str = ""

class UpdateStorytellerRequest(BaseModel):
    name: Optional[str] = None; system_prompt: Optional[str] = None; greeting: Optional[str] = None
    tone: Optional[str] = None; scripture: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Sutradhar API"}


# ── Metadata ───────────────────────────────────────────────────────────────────
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


# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def admin_login(request: LoginRequest):
    if not authenticate_admin(request.username, request.password):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": request.username, "role": "admin"})
    activity_log("login", f"Admin '{request.username}' logged in", actor=request.username)
    return {"access_token": token, "token_type": "bearer"}


# ── Ask ────────────────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_question(request: AskRequest):
    if request.scripture   not in SCRIPTURES:    raise HTTPException(400, f"Unknown scripture: {request.scripture}")
    if request.storyteller not in STORYTELLERS:  raise HTTPException(400, f"Unknown storyteller: {request.storyteller}")
    if request.language    not in LANGUAGE_CODES: raise HTTPException(400, f"Unsupported language: {request.language}")
    result = ask(query=request.question, scripture=request.scripture, storyteller=request.storyteller, language=request.language)
    if result["error"]: raise HTTPException(500, result["error"])
    return result

@app.post("/ask/voice")
async def ask_by_voice(audio: UploadFile = File(...), scripture: str = "ramayana", storyteller: str = "valmiki", language: str = "English"):
    audio_bytes = await audio.read()
    stt = speech_to_text(audio_bytes, language=language)
    if stt["error"]: raise HTTPException(500, f"STT error: {stt['error']}")
    if not stt["transcript"]: raise HTTPException(400, "Could not transcribe audio")
    result = ask(query=stt["transcript"], scripture=scripture, storyteller=storyteller, language=language)
    if result["error"]: raise HTTPException(500, result["error"])
    return {"transcript": stt["transcript"], **result}

@app.post("/tts")
def tts_endpoint(request: TTSRequest):
    import base64
    result = text_to_speech(request.text, language=request.language)
    if result["error"]: raise HTTPException(500, result["error"])
    return {"audio_base64": base64.b64encode(result["audio_bytes"]).decode(), "format": "wav"}


# ── Documents ──────────────────────────────────────────────────────────────────
@app.get("/documents")
def get_documents(scripture: str = None, search: str = None, page: int = 1, page_size: int = 20, _admin=Depends(require_admin)):
    docs = list_documents(scripture=scripture)
    if search:
        q = search.lower()
        docs = [d for d in docs if q in d.get("source","").lower() or q in d.get("kanda","").lower() or q in d.get("topic","").lower()]
    total = len(docs)
    start = (page - 1) * page_size
    return {"documents": docs[start:start+page_size], "total": total, "page": page, "page_size": page_size, "pages": max(1, (total + page_size - 1) // page_size)}

@app.get("/documents/export")
def export_documents(scripture: str = None, _admin=Depends(require_admin)):
    """Export all documents as JSON (admin downloads and converts to CSV client-side)."""
    docs = list_documents(scripture=scripture)
    return {"documents": docs, "total": len(docs)}

def _run_ingestion(job_id, tmp_path, ext, scripture, source, kanda, topic, filename, actor):
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
        activity_log("ingest_file", f"Ingested '{filename}' into {scripture} ({total} chunks)", actor=actor, meta={"doc_id": doc["id"], "chunks": total})
    except Exception as e:
        update_job(job_id, status="error", message=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), scripture: str = Form(default="ramayana"), source: str = Form(default=""), kanda: str = Form(default=""), topic: str = Form(default=""), admin=Depends(require_admin)):
    filename = file.filename or "upload"
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]: raise HTTPException(400, f"Unsupported file type: {ext}")
    if scripture not in SCRIPTURES:          raise HTTPException(400, f"Unknown scripture: {scripture}")
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content); tmp_path = tmp.name
    job    = create_job(filename=filename, scripture=scripture)
    actor  = admin.get("sub", "admin")
    thread = threading.Thread(target=_run_ingestion, args=(job["id"], tmp_path, ext, scripture, source or filename, kanda, topic, filename, actor), daemon=True)
    thread.start()
    return {"job_id": job["id"], "message": "Ingestion started", "filename": filename}

@app.get("/documents/jobs/{job_id}")
def get_job_status(job_id: str, _admin=Depends(require_admin)):
    job = get_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return job

@app.post("/documents/ingest-url")
def ingest_url(request: IngestURLRequest, admin=Depends(require_admin)):
    if request.scripture not in SCRIPTURES: raise HTTPException(400, f"Unknown scripture: {request.scripture}")
    try:
        text      = extract_text(request.url, is_url=True)
        chunks    = chunk_text(text)
        namespace = SCRIPTURES[request.scripture]["pinecone_namespace"]
        total     = upsert_to_pinecone(chunks=chunks, namespace=namespace, source=request.source or request.url, kanda=request.kanda, topic=request.topic)
        doc       = add_document(source=request.source or request.url, scripture=request.scripture, kanda=request.kanda, topic=request.topic, chunk_count=total, doc_type="url")
        activity_log("ingest_url", f"Ingested URL '{request.url}' into {request.scripture} ({total} chunks)", actor=admin.get("sub","admin"), meta={"doc_id": doc["id"]})
        return {"message": "URL ingested successfully", "document_id": doc["id"], "chunk_count": total}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.patch("/documents/{doc_id}")
def edit_document(doc_id: str, request: UpdateDocRequest, admin=Depends(require_admin)):
    doc = get_document(doc_id)
    if not doc: raise HTTPException(404, "Document not found")
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    updated = update_document(doc_id, updates)
    activity_log("edit_doc", f"Edited document '{doc['source']}'", actor=admin.get("sub","admin"), meta={"doc_id": doc_id})
    return {"message": "Document updated", "document": updated}

@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str, admin=Depends(require_admin)):
    doc = get_document(doc_id)
    if not doc: raise HTTPException(404, "Document not found")
    delete_document(doc_id)
    activity_log("delete_doc", f"Removed document '{doc['source']}'", actor=admin.get("sub","admin"), meta={"doc_id": doc_id})
    return {"message": "Document removed", "document_id": doc_id}


# ── Pinecone namespace management ──────────────────────────────────────────────
@app.delete("/namespaces/{namespace}")
def clear_namespace(namespace: str, admin=Depends(require_admin)):
    """Delete ALL vectors in a Pinecone namespace. Irreversible."""
    try:
        from pinecone import Pinecone
        pc    = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(os.environ.get("PINECONE_INDEX", "sutradhar"))
        index.delete(delete_all=True, namespace=namespace)
        activity_log("clear_namespace", f"Cleared Pinecone namespace '{namespace}'", actor=admin.get("sub","admin"))
        return {"message": f"Namespace '{namespace}' cleared successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Activity log ───────────────────────────────────────────────────────────────
@app.get("/admin/activity")
def get_activity(limit: int = 50, action: str = None, _admin=Depends(require_admin)):
    return {"activity": get_log(limit=limit, action_filter=action)}


# ── Admin user management ──────────────────────────────────────────────────────
@app.get("/admin/users")
def get_admin_users(_admin=Depends(require_admin)):
    env_admin = os.environ.get("ADMIN_USERNAME", "admin")
    users     = list_users()
    return {"users": users, "env_admin": env_admin}

@app.post("/admin/users")
def create_admin_user(request: CreateUserRequest, admin=Depends(require_admin)):
    try:
        user = create_user(request.username, request.password, request.display_name)
        activity_log("create_user", f"Created admin user '{request.username}'", actor=admin.get("sub","admin"))
        return {"message": "User created", "user": user}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.patch("/admin/users/{user_id}")
def update_admin_user(user_id: str, request: UpdateUserRequest, admin=Depends(require_admin)):
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    user    = update_user(user_id, updates)
    if not user: raise HTTPException(404, "User not found")
    activity_log("update_user", f"Updated admin user '{user['username']}'", actor=admin.get("sub","admin"))
    return {"message": "User updated", "user": user}

@app.delete("/admin/users/{user_id}")
def delete_admin_user(user_id: str, admin=Depends(require_admin)):
    if not delete_user(user_id): raise HTTPException(404, "User not found")
    activity_log("delete_user", f"Deleted admin user id '{user_id}'", actor=admin.get("sub","admin"))
    return {"message": "User deleted"}


# ── Scripture & storyteller management ────────────────────────────────────────
@app.get("/admin/scriptures")
def admin_list_scriptures(_admin=Depends(require_admin)):
    return {"scriptures": list_scriptures()}

@app.post("/admin/scriptures")
def admin_create_scripture(request: CreateScriptureRequest, admin=Depends(require_admin)):
    try:
        scripture = create_scripture(**request.model_dump())
        activity_log("create_scripture", f"Added scripture '{request.name}'", actor=admin.get("sub","admin"))
        return {"message": "Scripture created", "scripture": scripture}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.patch("/admin/scriptures/{scripture_id}")
def admin_update_scripture(scripture_id: str, request: UpdateScriptureRequest, admin=Depends(require_admin)):
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    updated = update_scripture(scripture_id, updates)
    if not updated: raise HTTPException(404, "Scripture not found")
    activity_log("update_scripture", f"Updated scripture '{scripture_id}'", actor=admin.get("sub","admin"))
    return {"message": "Scripture updated", "scripture": updated}

@app.get("/admin/storytellers")
def admin_list_storytellers(_admin=Depends(require_admin)):
    return {"storytellers": list_storytellers()}

@app.post("/admin/storytellers")
def admin_create_storyteller(request: CreateStorytellerRequest, admin=Depends(require_admin)):
    try:
        storyteller = create_storyteller(**request.model_dump())
        activity_log("create_storyteller", f"Added storyteller '{request.name}'", actor=admin.get("sub","admin"))
        return {"message": "Storyteller created", "storyteller": storyteller}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.patch("/admin/storytellers/{storyteller_id}")
def admin_update_storyteller(storyteller_id: str, request: UpdateStorytellerRequest, admin=Depends(require_admin)):
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    updated = update_storyteller(storyteller_id, updates)
    if not updated: raise HTTPException(404, "Storyteller not found")
    activity_log("update_storyteller", f"Updated storyteller '{storyteller_id}'", actor=admin.get("sub","admin"))
    return {"message": "Storyteller updated", "storyteller": updated}

@app.delete("/admin/storytellers/{storyteller_id}")
def admin_delete_storyteller(storyteller_id: str, admin=Depends(require_admin)):
    if not delete_storyteller(storyteller_id): raise HTTPException(404, "Storyteller not found")
    activity_log("delete_storyteller", f"Deleted storyteller '{storyteller_id}'", actor=admin.get("sub","admin"))
    return {"message": "Storyteller deleted"}


# ── Stats ──────────────────────────────────────────────────────────────────────
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