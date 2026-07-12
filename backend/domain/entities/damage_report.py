"""
Damage Report Entity
=====================
Represents the final damage assessment output —
the aggregated result combining classification, segmentation,
and detection into a severity score with recommendations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

from backend.domain.value_objects.severity import Severity


@dataclass
class ScoreComponent:
    """A single component of the severity score breakdown."""
    name: str
    raw_value: str
    score: float
    weight: float

    @property
    def contribution(self) -> float:
        """Weighted contribution to the total score."""
        return round(self.score * self.weight, 2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "score": round(self.score, 2),
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass
class Recommendation:
    """A single actionable recommendation from the assessment."""
    priority: str   # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO"
    action: str     # Human-readable action description

    def to_dict(self) -> dict:
        return {"priority": self.priority, "action": self.action}


@dataclass
class DamageReport:
    """
    The final assessment output — combines all pipeline results.

    Attributes:
        severity: Calculated damage severity
        flood_area_percentage: Percentage of image covered by flood
        total_objects: Total objects detected
        objects_in_flood: Objects detected within the flood zone
        objects_by_class: Detection counts grouped by class
        score_breakdown: How the severity score was calculated
        recommendations: List of prioritized action items
        generated_at: When this report was generated
    """

    severity: Severity
    flood_area_percentage: float = 0.0
    total_objects: int = 0
    objects_in_flood: int = 0
    objects_by_class: dict[str, int] = field(default_factory=dict)
    score_breakdown: list[ScoreComponent] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def category(self) -> str:
        """Damage category string."""
        return self.severity.category.value

    @property
    def category_emoji(self) -> str:
        """Emoji for the damage category."""
        return self.severity.emoji

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "severity": self.severity.to_dict(),
            "flood_area_percentage": self.flood_area_percentage,
            "total_objects": self.total_objects,
            "objects_in_flood": self.objects_in_flood,
            "objects_by_class": self.objects_by_class,
            "score_breakdown": [c.to_dict() for c in self.score_breakdown],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "generated_at": self.generated_at.isoformat(),
        }
