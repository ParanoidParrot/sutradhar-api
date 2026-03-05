"""
auth.py
JWT-based admin authentication for Sutradhar API.
Admin credentials are set via environment variables.
"""

import os
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "sutradhar-secret-change-in-production")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_HR = 24

ADMIN_USERNAME  = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD  = os.environ.get("ADMIN_PASSWORD", "sutradhar-admin")

pwd_context     = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme   = HTTPBearer()


# ── Password utils ────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

# Pre-hash the admin password at startup
ADMIN_PASSWORD_HASH = hash_password(ADMIN_PASSWORD[:72])


# ── Token utils ───────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HR)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# ── Admin login ───────────────────────────────────────────────────────────────
def authenticate_admin(username: str, password: str) -> bool:
    if username != ADMIN_USERNAME:
        return False
    return verify_password(password, ADMIN_PASSWORD_HASH)


# ── FastAPI dependency — protect admin endpoints ──────────────────────────────
def require_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    payload = verify_token(credentials.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return payload