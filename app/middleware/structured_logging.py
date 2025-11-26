"""
FastAPI middleware for structured logging request context.
"""
import asyncio
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging_config import bind_request_context, clear_request_context, get_logger

logger = get_logger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add structured logging context to each request.

    This middleware:
    - Generates a unique request_id for each request
    - Binds request context (request_id, method, path) to structured logs
    - Logs request completion/failure
    - Clears context after request completion
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add structured logging context.

        Parameters
        ----------
        request : Request
            The incoming request
        call_next : Callable
            The next middleware/handler in the chain

        Returns
        -------
        Response
            The response from the next handler
        """
        request_id = str(uuid4())
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        
        try:
            response = await call_next(request)
            logger.info(
                "Request completed",
                status_code=response.status_code,
                request_id=request_id,
            )
            return response
        except asyncio.CancelledError:
            # Don't log cancelled errors (normal during shutdown/client disconnect)
            # Re-raise immediately without logging
            raise
        except Exception as e:
            logger.error(
                "Request failed",
                exc_info=e,
                request_id=request_id,
            )
            raise
        finally:
            # Always clear context, even on cancellation
            clear_request_context()

