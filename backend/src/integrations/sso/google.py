from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth

from utils.config import settings

router = APIRouter(prefix="/auth/google", tags=["auth", "sso"])


def _get_oauth() -> OAuth:
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google SSO is not configured.")

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


@router.get("/login")
async def login(request: Request):
    oauth = _get_oauth()
    redirect_uri = request.url_for("google_auth_callback")
    return await oauth.google.authorize_redirect(request, str(redirect_uri))


@router.get("/callback")
async def google_auth_callback(request: Request):
    oauth = _get_oauth()
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo", {})

    request.state.sso_profile = user_info
    return RedirectResponse(url="/console")
