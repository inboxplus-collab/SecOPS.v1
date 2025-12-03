from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth

from utils.config import settings

router = APIRouter(prefix="/auth/okta", tags=["auth", "sso"])


def _get_oauth() -> OAuth:
    if not settings.OKTA_CLIENT_ID or not settings.OKTA_CLIENT_SECRET or not settings.OKTA_ISSUER_URL:
        raise HTTPException(status_code=500, detail="Okta SSO is not configured.")

    oauth = OAuth()
    oauth.register(
        name="okta",
        client_id=settings.OKTA_CLIENT_ID,
        client_secret=settings.OKTA_CLIENT_SECRET,
        server_metadata_url=f"{settings.OKTA_ISSUER_URL.rstrip('/')}/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth


@router.get("/login")
async def login(request: Request):
    oauth = _get_oauth()
    redirect_uri = request.url_for("okta_auth_callback")
    return await oauth.okta.authorize_redirect(request, str(redirect_uri))


@router.get("/callback")
async def okta_auth_callback(request: Request):
    oauth = _get_oauth()
    token = await oauth.okta.authorize_access_token(request)
    user_info = token.get("userinfo", {})

    request.state.sso_profile = user_info
    return RedirectResponse(url="/console")
