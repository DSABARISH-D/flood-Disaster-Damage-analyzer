"""
Health Check Routes
=====================
Liveness and readiness probes for monitoring and deployment.
"""

from fastapi import APIRouter

from backend.config.dependencies import get_classifier, get_segmenter, get_detector

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok", "service": "flood-damage-assessment-api"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — checks if all AI models are loaded.
    Returns 503 if any model is not ready.
    """
    try:
        models_status = {
            "classifier": get_classifier().is_loaded(),
            "segmenter": get_segmenter().is_loaded(),
            "detector": get_detector().is_loaded(),
        }
        all_ready = all(models_status.values())

        return {
            "status": "ready" if all_ready else "loading",
            "models": models_status,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
