"""
activity_store.py — Tracks admin activity log
Stores recent actions: ingestion, deletion, edits, namespace clears, user management
"""

import os
import json
from datetime import datetime

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ACTIVITY_FILE  = os.path.join(BASE_DIR, "data", "activity_log.json")
MAX_ENTRIES    = 200


def _load() -> list[dict]:
    if not os.path.exists(ACTIVITY_FILE):
        return []
    with open(ACTIVITY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(entries: list[dict]):
    os.makedirs(os.path.dirname(ACTIVITY_FILE), exist_ok=True)
    with open(ACTIVITY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def log(action: str, detail: str, actor: str = "admin", meta: dict = None):
    entries = _load()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "actor":     actor,
        "action":    action,   # e.g. "ingest_file", "delete_doc", "clear_namespace"
        "detail":    detail,
        "meta":      meta or {}
    }
    entries.insert(0, entry)
    _save(entries[:MAX_ENTRIES])
    return entry


def get_log(limit: int = 50, action_filter: str = None) -> list[dict]:
    entries = _load()
    if action_filter:
        entries = [e for e in entries if e["action"] == action_filter]
    return entries[:limit]