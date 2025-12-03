from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import AnyHttpUrl, Field

try:  # pragma: no cover - import guard for environments without pydantic-settings
    from pydantic_settings import BaseSettings
except Exception:  # noqa: BLE001
    from pydantic import BaseModel as BaseSettings  # type: ignore
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application runtime configuration with sensible defaults for tests."""

    # ========= GENERAL ==========
    APP_NAME: str = "SecOps AI"
    APP_VERSION: str = "0.1.0"
    ENV: str = Field(default="development")
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    # ========= DATABASE ==========
    DATABASE_URL: str = Field(default="sqlite:///:memory:")

    # ========= SUPABASE ==========
    SUPABASE_URL: Optional[AnyHttpUrl] = None
    SUPABASE_ANON_KEY: str = Field(default="anon-key")
    SUPABASE_SERVICE_KEY: str = Field(default="service-key")
    SUPABASE_JWT_SECRET: str = Field(default="test-secret")
    SUPABASE_KEY: str | None = None

    # ========= AUTH ==========
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # ========= SSO / OIDC ==========
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    OKTA_CLIENT_ID: Optional[str] = None
    OKTA_CLIENT_SECRET: Optional[str] = None
    OKTA_ISSUER_URL: Optional[str] = None
    AZURE_AD_CLIENT_ID: Optional[str] = None
    AZURE_AD_CLIENT_SECRET: Optional[str] = None
    AZURE_AD_TENANT_ID: Optional[str] = None

    # ========= GITHUB ==========
    GITHUB_APP_ID: str | None = None
    GITHUB_APP_PRIVATE_KEY: str | None = None
    GITHUB_WEBHOOK_SECRET: str | None = None

    # ========= LLM PROVIDERS ==========
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    EMERGENT_LLM_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    DEFAULT_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "groq/llama3-70b-8192"

    # ========= RAG ==========
    VECTOR_DB_URL: str | None = None
    VECTOR_DB_API_KEY: str | None = None
    VECTOR_STORE_TYPE: str = "mock"

    # ========= LOGGING / OBSERVABILITY ==========
    SENTRY_DSN: str | None = None
    ENABLE_OPENTELEMETRY: bool = False
    ENABLE_PROMETHEUS: bool = True

    # ========= KUBERNETES ==========
    K8S_IN_CLUSTER: bool = False
    K8S_KUBECONFIG: str | None = None

    # ========= BILLING ==========
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_PRICE_PRO: str | None = None
    STRIPE_PRICE_ENTERPRISE: str | None = None
    LEMONSQUEEZY_API_KEY: str | None = None
    FRONTEND_URL: str = Field(default="http://localhost:3000")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


@lru_cache(maxsize=1)
def validate_runtime_config() -> List[str]:
    """Best-effort validation that surfaces missing critical settings."""

    issues: List[str] = []

    if not settings.SUPABASE_JWT_SECRET:
        issues.append("SUPABASE_JWT_SECRET is not configured; JWT verification will fail.")

    if not settings.DATABASE_URL:
        issues.append("DATABASE_URL is not set; database connections will be disabled.")

    if not (settings.OPENAI_API_KEY or settings.GROQ_API_KEY or settings.MISTRAL_API_KEY):
        issues.append("No LLM provider keys configured; AI features will be disabled.")

    return issues


__all__ = ["settings", "validate_runtime_config", "Settings"]
