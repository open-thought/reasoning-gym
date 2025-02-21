"""API key middleware for FastAPI."""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to check for valid API key in request headers."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != self.api_key:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

        return await call_next(request)
