"""
document_store.py
Tracks ingested documents in a local JSON file.
Stores metadata like source, scripture, chunk count, date added.
"""

import os
import json
import uuid
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STORE_FILE = os.path.join(BASE_DIR, "data", "document_store.json")


def _load() -> list[dict]:
    if not os.path.exists(STORE_FILE):
        return []
    with open(STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(docs: list[dict]):
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)


def add_document(
    source:     str,
    scripture:  str,
    kanda:      str = "",
    topic:      str = "",
    chunk_count: int = 0,
    doc_type:   str = "file"
) -> dict:
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


def list_documents(scripture: str = None) -> list[dict]:
    docs = _load()
    if scripture:
        docs = [d for d in docs if d["scripture"] == scripture]
    return sorted(docs, key=lambda d: d["added_at"], reverse=True)


def delete_document(doc_id: str) -> bool:
    docs = _load()
    new_docs = [d for d in docs if d["id"] != doc_id]
    if len(new_docs) == len(docs):
        return False
    _save(new_docs)
    return True


def get_document(doc_id: str) -> dict | None:
    docs = _load()
    for d in docs:
        if d["id"] == doc_id:
            return d
    return None