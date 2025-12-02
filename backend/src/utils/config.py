from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl, Field

try:  # pragma: no cover - import guard for environments without pydantic-settings
    from pydantic_settings import BaseSettings
except Exception:  # noqa: BLE001
    from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    # Environment
    env: str = Field("development", description="Environment name: development/staging/production")

    # Backend
    app_name: str = "SecOpsAI Backend"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Database (Supabase Postgres in production)
    database_url: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/secops",
        description="SQLAlchemy async DB URL",
    )

    # Supabase
    supabase_url: Optional[AnyHttpUrl] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None

    # LLM / RAG
    rag_default_model: str = "gpt-4o-mini"  # or any other, your llm_client will interpret this
    rag_max_context_tokens: int = 4096
    rag_default_provider: str = "default"

    # Security
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    request_log_sample_rate: float = 0.1

    # Sentry
    sentry_dsn: Optional[str] = None
    sentry_traces_sample_rate: float = 0.1
    sentry_profiles_sample_rate: float = 0.0

    # Metrics
    enable_prometheus: bool = True

    # Vector store
    vector_store_type: str = "in_memory"

    # External provider keys (optional)
    emergent_llm_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    # Backwards compatible aliases for legacy code paths
    @property
    def APP_NAME(self) -> str:
        return self.app_name

    @property
    def APP_VERSION(self) -> str:
        return self.app_version

    @property
    def ENVIRONMENT(self) -> str:
        return self.env

    @property
    def DEBUG(self) -> bool:
        return self.env == "development"

    @property
    def DEFAULT_LLM_PROVIDER(self) -> str:
        return self.rag_default_provider

    @property
    def DEFAULT_LLM_MODEL(self) -> str:
        return self.rag_default_model

    @property
    def EMERGENT_LLM_KEY(self) -> Optional[str]:
        return self.emergent_llm_key

    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return self.openai_api_key

    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        return self.anthropic_api_key

    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return self.gemini_api_key

    @property
    def SUPABASE_URL(self) -> Optional[AnyHttpUrl]:
        return self.supabase_url

    @property
    def SUPABASE_KEY(self) -> Optional[str]:
        return self.supabase_anon_key

    @property
    def VECTOR_STORE_TYPE(self) -> str:
        return self.vector_store_type


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def validate_runtime_config(cfg: Settings | None = None) -> list[str]:
    """
    Validate critical runtime configuration and return human-friendly warnings.

    The goal is to surface missing environment variables early during startup
    instead of failing later during request handling.
    """

    cfg = cfg or settings
    issues: list[str] = []

    # Supabase auth relies on a JWT secret; warn if it is absent to avoid
    # accepting tokens without verification.
    if not cfg.supabase_jwt_secret:
        issues.append(
            "SUPABASE_JWT_SECRET is not configured; Supabase JWT verification will fail."
        )

    # Supabase client requires both URL and anon/service key to operate.
    if not cfg.supabase_url or not (
        cfg.supabase_anon_key or cfg.supabase_service_role_key
    ):
        issues.append(
            "Supabase URL or keys are missing; Supabase storage/DB helpers will be disabled."
        )

    llm_base = os.getenv("LLM_BASE_URL", "").strip()
    if not llm_base:
        issues.append(
            "LLM_BASE_URL is not set; RAG generation endpoints will return errors until configured."
        )

    return issues
