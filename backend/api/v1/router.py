"""
API v1 Router
===============
Aggregates all v1 route modules into a single router.
"""

from fastapi import APIRouter

from backend.api.v1.analysis_routes import router as analysis_router
from backend.api.v1.health_routes import router as health_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(analysis_router)
v1_router.include_router(health_router)
