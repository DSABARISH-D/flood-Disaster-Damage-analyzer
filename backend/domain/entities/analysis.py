"""
Analysis Entity
================
The central business object — represents a complete damage
analysis lifecycle from image upload to final assessment.

This is an ENTITY (not a Value Object) because:
  - Each Analysis has a unique `id`.
  - Two analyses with identical results but different IDs are DIFFERENT.
  - It has a lifecycle (PENDING → PROCESSING → COMPLETED).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.domain.enums.analysis_status import AnalysisStatus
from backend.domain.enums.damage_category import DamageCategory
from backend.domain.value_objects.severity import Severity
from backend.domain.value_objects.geo_coordinate import GeoCoordinate
from backend.domain.value_objects.image_metadata import ImageMetadata


@dataclass
class Analysis:
    """
    Root aggregate entity for a flood damage analysis.

    Lifecycle:
        1. Created with status=PENDING when user uploads an image.
        2. Transitions to PROCESSING when the AI pipeline starts.
        3. Each pipeline stage populates its result fields.
        4. Transitions to COMPLETED (success) or FAILED (error).

    Attributes:
        id: Unique identifier (UUID)
        user_id: ID of the user who created this analysis
        status: Current lifecycle status
        image_url: URL of the original uploaded image in storage
        image_metadata: Metadata about the uploaded image
        location: Optional GPS coordinates
        classification: Result from ResNet50 classifier
        segmentation: Result from SegFormer segmenter
        detections: List of detected objects from YOLOv8
        severity: Calculated damage severity
        recommendations: List of actionable recommendations
        annotated_image_url: URL of the image with detection overlays
        mask_image_url: URL of the flood mask overlay image
        error_message: Error description if status is FAILED
        created_at: Timestamp when the analysis was created
        updated_at: Timestamp of the last update
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""

    # Status
    status: AnalysisStatus = AnalysisStatus.PENDING

    # Input
    image_url: str = ""
    image_metadata: ImageMetadata | None = None
    location: GeoCoordinate | None = None

    # Pipeline results (populated progressively)
    classification: dict[str, Any] | None = None
    segmentation: dict[str, Any] | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)

    # Assessment
    severity: Severity | None = None
    damage_category: DamageCategory | None = None
    recommendations: list[dict[str, str]] = field(default_factory=list)
    score_breakdown: dict[str, Any] | None = None

    # Result images
    annotated_image_url: str = ""
    mask_image_url: str = ""

    # Error tracking
    error_message: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # ── Lifecycle Methods ──────────────────────────────

    def start_processing(self) -> None:
        """Mark this analysis as currently being processed."""
        self.status = AnalysisStatus.PROCESSING
        self._touch()

    def complete(self, severity: Severity, recommendations: list[dict]) -> None:
        """Mark this analysis as successfully completed."""
        self.status = AnalysisStatus.COMPLETED
        self.severity = severity
        self.damage_category = severity.category
        self.recommendations = recommendations
        self._touch()

    def fail(self, error_message: str) -> None:
        """Mark this analysis as failed with an error message."""
        self.status = AnalysisStatus.FAILED
        self.error_message = error_message
        self._touch()

    # ── Result Setters ─────────────────────────────────

    def set_classification(self, result: dict[str, Any]) -> None:
        """Store classification result from ResNet50."""
        self.classification = result
        self._touch()

    def set_segmentation(self, result: dict[str, Any]) -> None:
        """Store segmentation result from SegFormer."""
        self.segmentation = result
        self._touch()

    def set_detections(self, detections: list[dict[str, Any]]) -> None:
        """Store detection results from YOLOv8."""
        self.detections = detections
        self._touch()

    # ── Query Properties ───────────────────────────────

    @property
    def is_flood_detected(self) -> bool:
        """Whether the classifier detected a flood."""
        if self.classification is None:
            return False
        return self.classification.get("is_flood", False)

    @property
    def flood_area_percentage(self) -> float:
        """Flood area as a percentage (0-100)."""
        if self.segmentation is None:
            return 0.0
        return round(self.segmentation.get("flood_area_ratio", 0.0) * 100, 2)

    @property
    def total_detections(self) -> int:
        """Total number of detected objects."""
        return len(self.detections)

    @property
    def detections_in_flood(self) -> int:
        """Number of detected objects within the flood zone."""
        return sum(1 for d in self.detections if d.get("in_flood_zone", False))

    @property
    def is_complete(self) -> bool:
        """Whether this analysis has finished (successfully or with error)."""
        return self.status.is_terminal

    # ── Serialization ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for API responses / DB storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status.value,
            "image_url": self.image_url,
            "image_metadata": self.image_metadata.to_dict() if self.image_metadata else None,
            "location": self.location.to_dict() if self.location else None,
            "classification": self.classification,
            "segmentation": {
                k: v for k, v in (self.segmentation or {}).items()
                if k != "flood_mask"  # Don't serialize numpy array
            } if self.segmentation else None,
            "detections": self.detections,
            "severity": self.severity.to_dict() if self.severity else None,
            "damage_category": self.damage_category.value if self.damage_category else None,
            "recommendations": self.recommendations,
            "score_breakdown": self.score_breakdown,
            "annotated_image_url": self.annotated_image_url,
            "mask_image_url": self.mask_image_url,
            "flood_area_percentage": self.flood_area_percentage,
            "total_detections": self.total_detections,
            "detections_in_flood": self.detections_in_flood,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    # ── Private ────────────────────────────────────────

    def _touch(self) -> None:
        """Update the `updated_at` timestamp."""
        self.updated_at = datetime.utcnow()
