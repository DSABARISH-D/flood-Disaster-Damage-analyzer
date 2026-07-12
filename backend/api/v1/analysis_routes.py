"""
Analysis API Routes
=====================
HTTP endpoints for image analysis operations.

POST /api/v1/analysis      — Upload image and run full pipeline
GET  /api/v1/analysis/{id} — Fetch a completed analysis
GET  /api/v1/analysis      — List all analyses for the current user
DELETE /api/v1/analysis/{id} — Delete an analysis
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import Optional

from backend.api.schemas.analysis_schema import AnalysisResponse, AnalysisListResponse, AnalysisSummary
from backend.config.dependencies import get_pipeline_orchestrator, get_analysis_repository
from backend.domain.value_objects.geo_coordinate import GeoCoordinate

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("", response_model=AnalysisResponse, status_code=201)
async def create_analysis(
    file: UploadFile = File(..., description="Image file (JPG, PNG, TIFF)"),
    latitude: Optional[float] = Form(None, description="GPS latitude (-90 to 90)"),
    longitude: Optional[float] = Form(None, description="GPS longitude (-180 to 180)"),
    user_id: str = Form(default="anonymous", description="User ID"),
):
    """
    Upload an image and run the full AI damage analysis pipeline.

    The pipeline:
      1. Preprocesses the image (OpenCV)
      2. Classifies flood/no-flood (ResNet50)
      3. If flood: segments flood regions (SegFormer)
      4. Detects objects in flood zone (YOLOv8)
      5. Assesses damage severity
      6. Saves results to database

    Returns the complete analysis results.
    """
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Allowed: {allowed}")

    # Read file bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(image_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "File too large. Maximum size is 50MB.")

    # Build location
    location = None
    if latitude is not None and longitude is not None:
        try:
            location = GeoCoordinate(lat=latitude, lon=longitude)
        except ValueError as e:
            raise HTTPException(422, f"Invalid GPS coordinates: {e}")

    # Run pipeline
    try:
        orchestrator = get_pipeline_orchestrator()
        analysis = await orchestrator.run(
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
            user_id=user_id,
            location=location,
        )
        return AnalysisResponse(**analysis.to_dict())
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {type(e).__name__}: {str(e)}")


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Fetch a single analysis by ID."""
    repo = get_analysis_repository()
    analysis = await repo.find_by_id(analysis_id)

    if analysis is None:
        raise HTTPException(404, f"Analysis not found: {analysis_id}")

    return AnalysisResponse(**analysis.to_dict())


@router.get("", response_model=AnalysisListResponse)
async def list_analyses(
    user_id: str = Query(default="anonymous"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """List all analyses for a user with pagination."""
    repo = get_analysis_repository()
    offset = (page - 1) * limit

    analyses = await repo.list_by_user(user_id, limit=limit, offset=offset)
    total = await repo.count_by_user(user_id)

    items = [
        AnalysisSummary(
            id=a.id,
            status=a.status.value,
            damage_category=a.damage_category.value if a.damage_category else None,
            severity_score=a.severity.score if a.severity else None,
            flood_area_percentage=a.flood_area_percentage,
            total_detections=a.total_detections,
            image_url=a.image_url,
            created_at=a.created_at.isoformat(),
        )
        for a in analyses
    ]

    return AnalysisListResponse(items=items, total=total, page=page, limit=limit)


@router.delete("/{analysis_id}", status_code=204)
async def delete_analysis(analysis_id: str):
    """Delete an analysis by ID."""
    repo = get_analysis_repository()
    deleted = await repo.delete(analysis_id)
    if not deleted:
        raise HTTPException(404, f"Analysis not found: {analysis_id}")
