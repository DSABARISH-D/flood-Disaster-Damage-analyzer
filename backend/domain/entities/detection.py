"""
Detection Entity
=================
Represents a single object detected by YOLOv8 in the image.
"""

from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4

from backend.domain.enums.object_class import ObjectClass
from backend.domain.value_objects.bounding_box import BoundingBox


@dataclass
class Detection:
    """
    A single detected object in a flood image.

    Attributes:
        id: Unique identifier
        object_class: What type of object was detected
        confidence: Model confidence (0.0 – 1.0)
        bbox: Bounding box location in the image
        in_flood_zone: Whether this object overlaps with the flood mask
        risk_level: Derived risk assessment (Low / High / Critical)
        status: Human-readable status string
    """

    object_class: ObjectClass
    confidence: float
    bbox: BoundingBox
    in_flood_zone: bool = False
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    @property
    def risk_level(self) -> str:
        """
        Derived risk level based on object type and flood zone status.

        Rules:
          - Person in flood zone → CRITICAL
          - Vehicle/structure in flood zone → HIGH
          - Anything not in flood zone → LOW
        """
        if not self.in_flood_zone:
            return "Low"
        if self.object_class.is_person:
            return "Critical"
        return "High"

    @property
    def status(self) -> str:
        """Human-readable status of this detection."""
        if not self.in_flood_zone:
            return "Safe"
        if self.object_class.is_person:
            return "In Danger"
        return "Submerged/Damaged"

    @property
    def risk_weight(self) -> float:
        """
        Weighted risk contribution for damage scoring.
        Only objects in the flood zone contribute to the score.
        """
        if not self.in_flood_zone:
            return 0.0
        return self.object_class.risk_weight

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "class": self.object_class.value,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "center": list(self.bbox.center),
            "in_flood_zone": self.in_flood_zone,
            "risk_level": self.risk_level,
            "status": self.status,
        }
