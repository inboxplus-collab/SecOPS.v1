from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None


class UserProfile(BaseModel):
    id: str
    email: Optional[EmailStr] = None
    role: str = "authenticated"
    created_at: Optional[datetime] = None


class MeResponse(BaseModel):
    user: UserProfile
