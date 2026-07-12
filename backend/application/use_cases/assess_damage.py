"""
Assess Damage Use Case
========================
Combines classification + segmentation + detection outputs
into a weighted severity score with category and recommendations.

This is pure business logic — no frameworks, no DB, no AI models.
"""

from backend.domain.entities.detection import Detection
from backend.domain.entities.damage_report import DamageReport, ScoreComponent, Recommendation
from backend.domain.value_objects.severity import Severity
from backend.domain.interfaces.classifier_interface import ClassificationResult
from backend.domain.interfaces.segmenter_interface import SegmentationResult
from backend.domain.interfaces.detector_interface import DetectionResult
from backend.config.settings import settings


class AssessDamageUseCase:
    """
    Aggregates all pipeline outputs into a final damage assessment.

    Severity Score Formula (0-10):
      score = (flood_area_score × 0.40)
            + (structures_score × 0.30)
            + (type_score × 0.30)
    """

    def __init__(self):
        self.w_flood = settings.DAMAGE_WEIGHT_FLOOD_AREA
        self.w_structures = settings.DAMAGE_WEIGHT_STRUCTURES
        self.w_types = settings.DAMAGE_WEIGHT_OBJECT_TYPES

    def execute(
        self,
        classification: ClassificationResult,
        segmentation: SegmentationResult,
        detection: DetectionResult,
    ) -> DamageReport:
        """Run the damage assessment."""
        ratio = segmentation.flood_area_ratio
        detections = detection.detections

        # Score components
        flood_score = self._score_flood_area(ratio)
        struct_score = self._score_structures(detection.objects_in_flood, detection.total_objects)
        type_score = self._score_types(detections)

        # Weighted total
        total = min(
            flood_score * self.w_flood + struct_score * self.w_structures + type_score * self.w_types,
            10.0
        )

        severity = Severity(score=round(total, 2))

        # Build score breakdown
        breakdown = [
            ScoreComponent("Flood Area", f"{ratio * 100:.1f}%", flood_score, self.w_flood),
            ScoreComponent("Structures", str(detection.objects_in_flood), struct_score, self.w_structures),
            ScoreComponent("Object Types", "—", type_score, self.w_types),
        ]

        # Generate recommendations
        recs = self._recommendations(severity.category.value, ratio, detections)

        return DamageReport(
            severity=severity,
            flood_area_percentage=round(ratio * 100, 2),
            total_objects=detection.total_objects,
            objects_in_flood=detection.objects_in_flood,
            objects_by_class=detection.by_class,
            score_breakdown=breakdown,
            recommendations=recs,
        )

    # ── Scoring ────────────────────────────────────────

    @staticmethod
    def _score_flood_area(ratio: float) -> float:
        if ratio <= 0:
            return 0.0
        if ratio >= 0.8:
            return 10.0
        return min(10.0, (ratio ** 0.6) * 12.5)

    @staticmethod
    def _score_structures(affected: int, total: int) -> float:
        if total == 0:
            return 0.0
        return min(affected * 1.5 + (affected / total) * 3.0, 10.0)

    @staticmethod
    def _score_types(detections: list[Detection]) -> float:
        total = sum(d.risk_weight for d in detections)
        return min(total, 10.0)

    # ── Recommendations ────────────────────────────────

    @staticmethod
    def _recommendations(category: str, ratio: float, detections: list[Detection]) -> list[Recommendation]:
        recs: list[Recommendation] = []

        people = sum(1 for d in detections if d.object_class.is_person and d.in_flood_zone)
        vehicles = sum(1 for d in detections if d.object_class.is_vehicle and d.in_flood_zone)

        if people > 0:
            recs.append(Recommendation("CRITICAL", f"🚨 {people} person(s) in flood zone — initiate rescue."))
        if vehicles > 0:
            recs.append(Recommendation("HIGH", f"🚗 {vehicles} vehicle(s) submerged — deploy recovery."))
        if category in ("severe", "critical"):
            recs.append(Recommendation("HIGH", "🏚️ Severe flooding — consider area evacuation."))
            recs.append(Recommendation("HIGH", "🚧 Block affected roads to prevent entry."))
        if ratio > 0.5:
            recs.append(Recommendation("MEDIUM", "💧 >50% flood coverage — request drainage resources."))
        if category == "minor":
            recs.append(Recommendation("LOW", "📋 Minor flooding — continue monitoring."))
        if not recs:
            recs.append(Recommendation("INFO", "✅ No immediate dangers. Continue monitoring."))

        return recs
