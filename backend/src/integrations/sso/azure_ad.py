from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth

from utils.config import settings

router = APIRouter(prefix="/auth/azure", tags=["auth", "sso"])


def _get_oauth() -> OAuth:
    if (
        not settings.AZURE_AD_CLIENT_ID
        or not settings.AZURE_AD_CLIENT_SECRET
        or not settings.AZURE_AD_TENANT_ID
    ):
        raise HTTPException(status_code=500, detail="Azure AD SSO is not configured.")

    metadata_url = (
        f"https://login.microsoftonline.com/{settings.AZURE_AD_TENANT_ID}/v2.0/.well-known/openid-configuration"
    )

    oauth = OAuth()
    oauth.register(
        name="azure",
        client_id=settings.AZURE_AD_CLIENT_ID,
        client_secret=settings.AZURE_AD_CLIENT_SECRET,
        server_metadata_url=metadata_url,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


@router.get("/login")
async def login(request: Request):
    oauth = _get_oauth()
    redirect_uri = request.url_for("azure_auth_callback")
    return await oauth.azure.authorize_redirect(request, str(redirect_uri))


@router.get("/callback")
async def azure_auth_callback(request: Request):
    oauth = _get_oauth()
    token = await oauth.azure.authorize_access_token(request)
    user_info = token.get("userinfo", {})

    request.state.sso_profile = user_info
    return RedirectResponse(url="/console")
