"""
🌊 Flood Damage Assessment API — FastAPI Entry Point
=====================================================
Creates and configures the FastAPI application.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1.router import v1_router
from backend.api.middleware.error_handler import global_exception_handler, value_error_handler
from backend.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: Pre-loads AI models so the first request is fast.
    Shutdown: Cleanup resources.
    """
    # ── Startup ────────────────────────────────────────
    print("🌊 Starting Flood Damage Assessment API...")
    print(f"   Device: {settings.DEVICE}")
    print(f"   Debug: {settings.DEBUG}")

    # Pre-warm models (lazy-loaded on first call)
    try:
        from backend.config.dependencies import get_classifier, get_segmenter, get_detector
        print("   Loading ResNet50 classifier...")
        get_classifier()
        print("   Loading SegFormer segmenter...")
        get_segmenter()
        print("   Loading YOLOv8 detector...")
        get_detector()
        print("   ✅ All models loaded!")
    except Exception as e:
        print(f"   ⚠️ Model loading warning: {e}")
        print("   Models will be loaded on first request.")

    yield

    # ── Shutdown ───────────────────────────────────────
    print("🌊 Shutting down API...")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-powered flood damage assessment pipeline using "
            "ResNet50, SegFormer, and YOLOv8 for classification, "
            "segmentation, and object detection."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS Middleware ────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ─────────────────────────────
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)

    # ── Routers ────────────────────────────────────────
    app.include_router(v1_router)

    # ── Root Redirect ──────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create the app instance
app = create_app()
