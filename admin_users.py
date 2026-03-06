"""
admin_users.py — Multi-admin user management
Stores hashed passwords in data/admin_users.json
Primary env-var admin always takes precedence.
"""

import os
import json
import uuid
from datetime import datetime
from passlib.context import CryptContext

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "data", "admin_users.json")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _load() -> list[dict]:
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(users: list[dict]):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def list_users() -> list[dict]:
    """Return users without password hashes."""
    return [
        {k: v for k, v in u.items() if k != "password_hash"}
        for u in _load()
    ]


def get_user(username: str) -> dict | None:
    for u in _load():
        if u["username"] == username:
            return u
    return None


def create_user(username: str, password: str, display_name: str = "") -> dict:
    users = _load()
    if any(u["username"] == username for u in users):
        raise ValueError(f"User '{username}' already exists")
    user = {
        "id":            str(uuid.uuid4()),
        "username":      username,
        "display_name":  display_name or username,
        "password_hash": pwd_context.hash(password[:72]),
        "created_at":    datetime.utcnow().isoformat(),
        "active":        True
    }
    users.append(user)
    _save(users)
    return {k: v for k, v in user.items() if k != "password_hash"}


def update_user(user_id: str, updates: dict) -> dict | None:
    users = _load()
    for i, u in enumerate(users):
        if u["id"] == user_id:
            if "password" in updates:
                updates["password_hash"] = pwd_context.hash(updates.pop("password")[:72])
            allowed = {"display_name", "password_hash", "active"}
            users[i] = {**u, **{k: v for k, v in updates.items() if k in allowed}}
            _save(users)
            return {k: v for k, v in users[i].items() if k != "password_hash"}
    return None


def delete_user(user_id: str) -> bool:
    users     = _load()
    new_users = [u for u in users if u["id"] != user_id]
    if len(new_users) == len(users):
        return False
    _save(new_users)
    return True


def verify_user(username: str, password: str) -> bool:
    user = get_user(username)
    if not user or not user.get("active"):
        return False
    return pwd_context.verify(password[:72], user["password_hash"])