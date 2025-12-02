# backend/src/main.py

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# Routers
from api.routes import platform, rag  # core routes used in tests

try:  # Optional routes may rely on heavyweight dependencies
    from api.routes import analysis, integrations, admin, auth  # type: ignore[attr-defined]
except Exception as exc:  # noqa: BLE001
    logging.getLogger(__name__).warning("Optional routers could not be loaded: %s", exc)
    analysis = integrations = admin = auth = None

from utils.config import validate_runtime_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Sentry – error monitoring (optional, enabled only if SENTRY_DSN is set)
# ---------------------------------------------------------------------------

SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()

if SENTRY_DSN:
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            FastApiIntegration(),
            sentry_logging,
        ],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
        environment=os.getenv("SENTRY_ENV", "local"),
        send_default_pii=False,
    )
    logger.info("Sentry initialized")
else:
    logger.info("Sentry DSN not configured; Sentry disabled")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SecOpsAI Backend",
    version="0.1.0",
    description="SecOpsAI – Autonomous DevSecOps intelligence backend API.",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "secops_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "secops_request_latency_seconds",
    "Request latency (seconds)",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "secops_errors_total",
    "Total number of exceptions",
    ["endpoint", "error_type"],
)

JOB_DURATION = Histogram(
    "secops_job_duration_seconds",
    "Duration of background/scheduled jobs",
    ["job_name"],
)

ACTIVE_JOBS = Gauge(
    "secops_active_jobs",
    "Number of currently running background jobs",
)


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """
    Middleware that records request counts and latencies for Prometheus.
    """
    start_time = time.time()
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:  # noqa: BLE001
        ERROR_COUNT.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        raise

    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    return response


@app.get("/metrics")
def get_metrics() -> Response:
    """
    Prometheus scrape endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


@app.get("/health", tags=["platform"])
async def health() -> Dict[str, Any]:
    """
    Simple health endpoint used by load balancers / uptime checks.
    """
    return {"status": "ok", "service": "secops-backend"}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# IMPORTANT: these assumes your router modules expose `router: APIRouter`

if auth:
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
if analysis:
    app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(rag.router)
app.include_router(platform.router)
if integrations:
    app.include_router(integrations.router, prefix="/api/integrations", tags=["integrations"])
if admin:
    app.include_router(admin.router, prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Startup / shutdown hooks (optional scheduler wiring)
# ---------------------------------------------------------------------------

try:
    from core.scheduler.scheduler import start_scheduler, shutdown_scheduler  # type: ignore
except Exception:  # noqa: BLE001
    start_scheduler = None
    shutdown_scheduler = None
    logger.info("Scheduler not available; skipping automatic startup.")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("SecOpsAI backend starting up")

    config_issues = validate_runtime_config()
    for issue in config_issues:
        logger.warning("Runtime configuration issue: %s", issue)
    if not config_issues:
        logger.info("Runtime configuration validated")

    if callable(start_scheduler):
        try:
            start_scheduler()
            logger.info("Scheduler started")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start scheduler: %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("SecOpsAI backend shutting down")
    if callable(shutdown_scheduler):
        try:
            shutdown_scheduler()
            logger.info("Scheduler shut down")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to shut down scheduler: %s", exc)
