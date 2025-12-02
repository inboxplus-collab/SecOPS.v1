# backend/src/utils/errors.py

from fastapi import HTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR


class SecOpsError(HTTPException):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=message)


class IntegrationError(SecOpsError):
    pass


class LLMError(SecOpsError):
    pass


class AnalyzerError(SecOpsError):
    pass


def internal_error(message: str = "Internal server error"):
    raise HTTPException(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        detail=message
    )
