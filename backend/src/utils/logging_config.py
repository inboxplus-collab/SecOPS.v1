# backend/src/utils/logging_config.py

import logging
import sys
import json
from backend.src.utils.config import settings

def setup_logging():
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log = {
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": self.formatTime(record),
                "logger": record.name,
            }
            return json.dumps(log)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.ERROR,
            event_level=logging.ERROR
        )
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=0.2,
            integrations=[sentry_logging]
        )
