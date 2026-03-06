"""
config_store.py — Manages scriptures.json and storytellers.json
Allows adding/editing/deactivating scriptures and storytellers from admin.
"""

import os
import json

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
SCRIPTURES_FILE   = os.path.join(BASE_DIR, "scriptures.json")
STORYTELLERS_FILE = os.path.join(BASE_DIR, "storytellers.json")


# ── Scriptures ────────────────────────────────────────────────────────────────
def _load_scriptures() -> list[dict]:
    with open(SCRIPTURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["scriptures"]

def _save_scriptures(items: list[dict]):
    with open(SCRIPTURES_FILE, "w", encoding="utf-8") as f:
        json.dump({"scriptures": items}, f, indent=2, ensure_ascii=False)

def list_scriptures(active_only=False) -> list[dict]:
    items = _load_scriptures()
    return [s for s in items if s.get("active")] if active_only else items

def get_scripture(scripture_id: str) -> dict | None:
    for s in _load_scriptures():
        if s["id"] == scripture_id:
            return s
    return None

def create_scripture(id, name, description, pinecone_namespace, default_storyteller, available_storytellers=None) -> dict:
    items = _load_scriptures()
    if any(s["id"] == id for s in items):
        raise ValueError(f"Scripture '{id}' already exists")
    item = {
        "id": id, "name": name, "description": description,
        "pinecone_namespace": pinecone_namespace,
        "default_storyteller": default_storyteller,
        "available_storytellers": available_storytellers or [default_storyteller],
        "active": True
    }
    items.append(item)
    _save_scriptures(items)
    return item

def update_scripture(scripture_id: str, updates: dict) -> dict | None:
    items = _load_scriptures()
    for i, s in enumerate(items):
        if s["id"] == scripture_id:
            allowed = {"name", "description", "pinecone_namespace", "default_storyteller", "available_storytellers", "active"}
            items[i] = {**s, **{k: v for k, v in updates.items() if k in allowed}}
            _save_scriptures(items)
            return items[i]
    return None


# ── Storytellers ──────────────────────────────────────────────────────────────
def _load_storytellers() -> list[dict]:
    with open(STORYTELLERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["storytellers"]

def _save_storytellers(items: list[dict]):
    with open(STORYTELLERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"storytellers": items}, f, indent=2, ensure_ascii=False)

def list_storytellers() -> list[dict]:
    return _load_storytellers()

def get_storyteller(storyteller_id: str) -> dict | None:
    for s in _load_storytellers():
        if s["id"] == storyteller_id:
            return s
    return None

def create_storyteller(id, name, scripture, system_prompt, greeting, tone) -> dict:
    items = _load_storytellers()
    if any(s["id"] == id for s in items):
        raise ValueError(f"Storyteller '{id}' already exists")
    item = {"id": id, "name": name, "scripture": scripture,
            "system_prompt": system_prompt, "greeting": greeting, "tone": tone}
    items.append(item)
    _save_storytellers(items)
    return item

def update_storyteller(storyteller_id: str, updates: dict) -> dict | None:
    items = _load_storytellers()
    for i, s in enumerate(items):
        if s["id"] == storyteller_id:
            allowed = {"name", "system_prompt", "greeting", "tone", "scripture"}
            items[i] = {**s, **{k: v for k, v in updates.items() if k in allowed}}
            _save_storytellers(items)
            return items[i]
    return None

def delete_storyteller(storyteller_id: str) -> bool:
    items     = _load_storytellers()
    new_items = [s for s in items if s["id"] != storyteller_id]
    if len(new_items) == len(items):
        return False
    _save_storytellers(new_items)
    return True