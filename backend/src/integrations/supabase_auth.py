# backend/src/integrations/supabase_auth.py

import jwt
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from backend.src.utils.config import settings


auth_scheme = HTTPBearer()

async def verify_supabase_jwt(request: Request):
    token = None

    # 1. Authorization header
    if "authorization" in request.headers:
        try:
            token = request.headers["authorization"].split(" ")[1]
        except:
            pass

    # 2. Cookie fallback
    if not token:
        token = request.cookies.get("sb-access-token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        request.state.user = payload
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Supabase JWT")
