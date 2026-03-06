"""
document_store.py
Tracks ingested documents and ingestion jobs in local JSON files.
"""

import os
import json
import uuid
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STORE_FILE = os.path.join(BASE_DIR, "data", "document_store.json")
JOBS_FILE  = os.path.join(BASE_DIR, "data", "jobs_store.json")


# ── Documents ─────────────────────────────────────────────────────────────────
def _load() -> list[dict]:
    if not os.path.exists(STORE_FILE):
        return []
    with open(STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(docs: list[dict]):
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)


def add_document(source, scripture, kanda="", topic="", chunk_count=0, doc_type="file") -> dict:
    docs = _load()
    doc  = {
        "id":          str(uuid.uuid4()),
        "source":      source,
        "scripture":   scripture,
        "kanda":       kanda,
        "topic":       topic,
        "chunk_count": chunk_count,
        "doc_type":    doc_type,
        "added_at":    datetime.utcnow().isoformat()
    }
    docs.append(doc)
    _save(docs)
    return doc


def list_documents(scripture=None) -> list[dict]:
    docs = _load()
    if scripture:
        docs = [d for d in docs if d["scripture"] == scripture]
    return sorted(docs, key=lambda d: d["added_at"], reverse=True)


def delete_document(doc_id: str) -> bool:
    docs     = _load()
    new_docs = [d for d in docs if d["id"] != doc_id]
    if len(new_docs) == len(docs):
        return False
    _save(new_docs)
    return True


def get_document(doc_id: str) -> dict | None:
    for d in _load():
        if d["id"] == doc_id:
            return d
    return None


def update_document(doc_id: str, updates: dict) -> dict | None:
    docs = _load()
    for i, d in enumerate(docs):
        if d["id"] == doc_id:
            allowed = {"source", "kanda", "topic", "scripture"}
            docs[i] = {**d, **{k: v for k, v in updates.items() if k in allowed}}
            _save(docs)
            return docs[i]
    return None


# ── Ingestion jobs ────────────────────────────────────────────────────────────
def _load_jobs() -> dict:
    if not os.path.exists(JOBS_FILE):
        return {}
    with open(JOBS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_jobs(jobs: dict):
    os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def create_job(filename: str, scripture: str) -> dict:
    jobs   = _load_jobs()
    job_id = str(uuid.uuid4())
    job    = {
        "id":         job_id,
        "filename":   filename,
        "scripture":  scripture,
        "status":     "pending",   # pending | processing | done | error
        "progress":   0,           # 0-100
        "message":    "Starting ingestion...",
        "chunk_count": 0,
        "doc_id":     None,
        "created_at": datetime.utcnow().isoformat()
    }
    jobs[job_id] = job
    _save_jobs(jobs)
    return job


def update_job(job_id: str, **kwargs) -> dict | None:
    jobs = _load_jobs()
    if job_id not in jobs:
        return None
    jobs[job_id] = {**jobs[job_id], **kwargs}
    _save_jobs(jobs)
    return jobs[job_id]


def get_job(job_id: str) -> dict | None:
    return _load_jobs().get(job_id)