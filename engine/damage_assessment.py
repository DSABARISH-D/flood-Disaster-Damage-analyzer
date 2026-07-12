"""
Damage Assessment Engine
=========================
Aggregates outputs from the classifier, segmenter, and detector
to produce a unified damage assessment with severity scoring.
"""

import numpy as np
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DamageAssessmentEngine:
    """
    Combines flood classification, segmentation, and object detection
    results into a comprehensive damage assessment report.
    """

    def __init__(self):
        self.severity_thresholds = config.DAMAGE_SEVERITY_THRESHOLDS
        self.weight_flood = config.DAMAGE_WEIGHT_FLOOD_AREA
        self.weight_structures = config.DAMAGE_WEIGHT_AFFECTED_STRUCTURES
        self.weight_types = config.DAMAGE_WEIGHT_OBJECT_TYPES

    def assess(
        self,
        classification_result: dict,
        segmentation_result: dict,
        detection_result: dict,
        location: dict = None,
    ) -> dict:
        """
        Perform comprehensive damage assessment.

        Args:
            classification_result: Output from FloodClassifier.predict()
            segmentation_result: Output from FloodSegmenter.segment()
            detection_result: Output from ObjectDetector.detect()
            location: Optional dict {"lat": float, "lon": float}

        Returns:
            dict: Complete damage assessment report
        """
        # ── Extract key metrics ────────────────────────────
        flood_area_ratio = segmentation_result.get("flood_area_ratio", 0.0)
        detections = detection_result.get("detections", [])
        summary = detection_result.get("summary", {})
        total_objects = summary.get("total_objects", 0)
        objects_in_flood = summary.get("objects_in_flood", 0)
        class_counts = summary.get("by_class", {})

        # ── Calculate severity components ──────────────────

        # 1. Flood area score (0-10)
        flood_area_score = self._score_flood_area(flood_area_ratio)

        # 2. Affected structures score (0-10)
        structures_score = self._score_affected_structures(objects_in_flood, total_objects)

        # 3. Object type severity score (0-10)
        type_score = self._score_object_types(detections)

        # ── Weighted severity calculation ──────────────────
        severity_score = (
            flood_area_score * self.weight_flood
            + structures_score * self.weight_structures
            + type_score * self.weight_types
        )
        severity_score = round(min(severity_score, 10.0), 2)

        # ── Determine damage category ─────────────────────
        damage_category = self._get_damage_category(severity_score)

        # ── Per-object damage assessment ───────────────────
        object_assessments = self._assess_objects(detections)

        # ── Build report ───────────────────────────────────
        report = {
            "timestamp": datetime.now().isoformat(),
            "location": location,

            # Classification
            "flood_detected": classification_result.get("is_flood", False),
            "flood_confidence": classification_result.get("confidence", 0.0),

            # Segmentation metrics
            "flood_area_percentage": round(flood_area_ratio * 100, 2),
            "flood_area_score": round(flood_area_score, 2),

            # Detection metrics
            "total_objects_detected": total_objects,
            "objects_in_flood_zone": objects_in_flood,
            "objects_by_class": class_counts,
            "structures_score": round(structures_score, 2),
            "type_score": round(type_score, 2),

            # Overall assessment
            "severity_score": severity_score,
            "damage_category": damage_category,
            "damage_category_emoji": self._get_category_emoji(damage_category),

            # Detailed assessments
            "object_assessments": object_assessments,

            # Recommendations
            "recommendations": self._generate_recommendations(
                damage_category, flood_area_ratio, detections
            ),

            # Score breakdown
            "score_breakdown": {
                "flood_area": {
                    "raw": round(flood_area_ratio * 100, 2),
                    "score": round(flood_area_score, 2),
                    "weight": self.weight_flood,
                    "contribution": round(flood_area_score * self.weight_flood, 2),
                },
                "affected_structures": {
                    "count": objects_in_flood,
                    "score": round(structures_score, 2),
                    "weight": self.weight_structures,
                    "contribution": round(structures_score * self.weight_structures, 2),
                },
                "object_types": {
                    "score": round(type_score, 2),
                    "weight": self.weight_types,
                    "contribution": round(type_score * self.weight_types, 2),
                },
            },
        }

        return report

    # ── Scoring Functions ──────────────────────────────────

    def _score_flood_area(self, ratio: float) -> float:
        """
        Convert flood area ratio (0-1) to a severity score (0-10).
        Non-linear scaling: small floods score low, large floods score exponentially.
        """
        if ratio <= 0:
            return 0.0
        if ratio >= 0.8:
            return 10.0
        # Exponential curve: accelerates as flood coverage increases
        return min(10.0, (ratio ** 0.6) * 12.5)

    def _score_affected_structures(self, affected: int, total: int) -> float:
        """Score based on number and ratio of affected objects."""
        if total == 0:
            return 0.0

        # Base score from count
        count_score = min(affected * 1.5, 7.0)

        # Ratio bonus
        ratio = affected / total
        ratio_bonus = ratio * 3.0

        return min(count_score + ratio_bonus, 10.0)

    def _score_object_types(self, detections: list) -> float:
        """
        Score based on the types of objects affected.
        People in danger > structures > vehicles
        """
        if not detections:
            return 0.0

        type_weights = {
            "person": 4.0,   # Highest priority
            "boat": 1.5,     # Indicates rescue operations
            "car": 2.0,
            "truck": 2.5,
            "bus": 3.0,      # Public transport = more people
            "bicycle": 1.0,
            "motorcycle": 1.5,
        }

        total_weight = 0.0
        for det in detections:
            if det.get("in_flood_zone", False):
                cls = det["class"]
                total_weight += type_weights.get(cls, 1.0)

        return min(total_weight, 10.0)

    def _get_damage_category(self, score: float) -> str:
        """Map severity score to damage category."""
        for category, (low, high) in self.severity_thresholds.items():
            if low <= score < high:
                return category.title()
        return "Critical"

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for damage category."""
        emoji_map = {
            "Minor": "🟡",
            "Moderate": "🟠",
            "Severe": "🔴",
            "Critical": "⚫",
        }
        return emoji_map.get(category, "⚪")

    # ── Object-level Assessment ────────────────────────────

    def _assess_objects(self, detections: list) -> list:
        """Produce per-object damage status."""
        assessments = []
        for det in detections:
            status = "Safe"
            risk_level = "Low"

            if det.get("in_flood_zone", False):
                status = "At Risk"
                risk_level = "High"
                if det["class"] == "person":
                    status = "In Danger"
                    risk_level = "Critical"
                elif det["class"] in config.YOLO_STRUCTURE_CLASSES:
                    status = "Submerged/Damaged"
                    risk_level = "High"

            assessments.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "status": status,
                "risk_level": risk_level,
                "in_flood_zone": det.get("in_flood_zone", False),
            })

        return assessments

    # ── Recommendations ────────────────────────────────────

    def _generate_recommendations(
        self, category: str, flood_ratio: float, detections: list
    ) -> list:
        """Generate actionable recommendations based on the assessment."""
        recs = []

        # People in danger
        people_at_risk = sum(
            1 for d in detections
            if d["class"] == "person" and d.get("in_flood_zone", False)
        )
        if people_at_risk > 0:
            recs.append({
                "priority": "CRITICAL",
                "action": f"🚨 {people_at_risk} person(s) detected in flood zone — initiate immediate rescue operations.",
            })

        # Vehicles trapped
        vehicles_trapped = sum(
            1 for d in detections
            if d["class"] in ("car", "truck", "bus", "motorcycle")
            and d.get("in_flood_zone", False)
        )
        if vehicles_trapped > 0:
            recs.append({
                "priority": "HIGH",
                "action": f"🚗 {vehicles_trapped} vehicle(s) detected in flood zone — deploy tow/rescue vehicles.",
            })

        # General severity-based
        if category in ("Severe", "Critical"):
            recs.append({
                "priority": "HIGH",
                "action": "🏚️ Severe flooding detected — consider evacuation of nearby residents.",
            })
            recs.append({
                "priority": "HIGH",
                "action": "🚧 Block off affected roads to prevent vehicle entry.",
            })

        if flood_ratio > 0.5:
            recs.append({
                "priority": "MEDIUM",
                "action": "💧 Over 50% flood coverage — request additional drainage resources.",
            })

        if category == "Minor":
            recs.append({
                "priority": "LOW",
                "action": "📋 Minor flooding detected — continue monitoring water levels.",
            })

        if not recs:
            recs.append({
                "priority": "INFO",
                "action": "✅ No immediate dangers detected. Continue standard monitoring.",
            })

        return recs
