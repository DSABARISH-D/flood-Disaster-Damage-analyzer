"""
Analysis Pydantic Schemas
===========================
Request/response validation models for the Analysis API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class AnalysisResponse(BaseModel):
    """Full analysis result returned by the API."""
    id: str
    user_id: str = ""
    status: str
    image_url: str = ""
    image_metadata: dict[str, Any] | None = None
    location: dict[str, float] | None = None

    # Classification
    classification: dict[str, Any] | None = None

    # Segmentation
    segmentation: dict[str, Any] | None = None
    flood_area_percentage: float = 0.0

    # Detection
    detections: list[dict[str, Any]] = []
    total_detections: int = 0
    detections_in_flood: int = 0

    # Assessment
    severity: dict[str, Any] | None = None
    damage_category: str | None = None
    recommendations: list[dict[str, str]] = []
    score_breakdown: dict[str, Any] | None = None

    # Result images
    annotated_image_url: str = ""
    mask_image_url: str = ""

    # Meta
    error_message: str = ""
    created_at: str = ""
    updated_at: str = ""

    model_config = {"from_attributes": True}


class AnalysisSummary(BaseModel):
    """Compact analysis summary for list endpoints."""
    id: str
    status: str
    damage_category: str | None = None
    severity_score: float | None = None
    flood_area_percentage: float = 0.0
    total_detections: int = 0
    image_url: str = ""
    created_at: str = ""


class AnalysisListResponse(BaseModel):
    """Paginated list of analysis summaries."""
    items: list[AnalysisSummary]
    total: int
    page: int
    limit: int
