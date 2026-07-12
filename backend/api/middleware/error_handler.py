"""
Global Error Handler Middleware
=================================
Catches unhandled exceptions and returns clean JSON error responses.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import traceback


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert unhandled exceptions to structured JSON error responses."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {type(exc).__name__}",
            "message": str(exc),
        },
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors with 422 status."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "message": str(exc)},
    )
