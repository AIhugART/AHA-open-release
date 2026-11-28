"""One error type for the whole API so every failure has the shape the spec promises."""

from __future__ import annotations


class ApiError(Exception):
    """An error that maps to an HTTP status and the /v1 error body (docs/product/03_api_v1.md section 8)."""

    def __init__(self, status: int, code: str, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "retry_after_s": self.retry_after_s}}


def invalid(message: str) -> ApiError:
    return ApiError(400, "invalid_request", message)


def not_found(what: str = "not found") -> ApiError:
    return ApiError(404, "not_found", what)


def unauthorized() -> ApiError:
    return ApiError(401, "unauthorized", "missing or invalid token")


def forbidden(message: str) -> ApiError:
    return ApiError(403, "forbidden", message)


def busy(message: str, retry_after_s: float = 5.0) -> ApiError:
    return ApiError(429, "busy", message, retry_after_s)
