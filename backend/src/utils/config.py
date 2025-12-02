# backend/src/utils/config.py

import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # ========= GENERAL ==========
    APP_NAME: str = "SecOps AI"
    ENV: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    # ========= DATABASE ==========
    DATABASE_URL: str

    # ========= SUPABASE ==========
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str

    SUPABASE_JWT_SECRET: str

    # ========= AUTH ==========
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # ========= GITHUB ==========
    GITHUB_APP_ID: str | None = None
    GITHUB_APP_PRIVATE_KEY: str | None = None
    GITHUB_WEBHOOK_SECRET: str | None = None

    # ========= LLM PROVIDERS ==========
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None

    DEFAULT_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "groq/llama3-70b-8192"

    # ========= RAG ==========
    VECTOR_DB_URL: str | None = None
    VECTOR_DB_API_KEY: str | None = None

    # ========= LOGGING / OBSERVABILITY ==========
    SENTRY_DSN: str | None = None
    ENABLE_OPENTELEMETRY: bool = False
    ENABLE_PROMETHEUS: bool = True

    # ========= KUBERNETES ==========
    K8S_IN_CLUSTER: bool = False
    K8S_KUBECONFIG: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
