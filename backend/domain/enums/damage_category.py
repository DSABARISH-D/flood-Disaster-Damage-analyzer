"""
Damage Category Enum
=====================
Classifies the overall severity of flood damage into
human-readable categories with associated visual indicators.
"""

from enum import Enum


class DamageCategory(str, Enum):
    """
    Damage severity classification.

    Each category maps to a score range (0-10):
      MINOR:    0.0 – 2.5
      MODERATE: 2.5 – 5.0
      SEVERE:   5.0 – 7.5
      CRITICAL: 7.5 – 10.0
    """

    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        """Visual emoji indicator for the category."""
        return {
            DamageCategory.MINOR: "🟡",
            DamageCategory.MODERATE: "🟠",
            DamageCategory.SEVERE: "🔴",
            DamageCategory.CRITICAL: "⚫",
        }[self]

    @property
    def color_hex(self) -> str:
        """CSS-friendly hex color for the category."""
        return {
            DamageCategory.MINOR: "#ffc107",
            DamageCategory.MODERATE: "#ff9800",
            DamageCategory.SEVERE: "#f44336",
            DamageCategory.CRITICAL: "#6a1b9a",
        }[self]

    @property
    def score_range(self) -> tuple[float, float]:
        """The score range (inclusive lower, exclusive upper) for this category."""
        return {
            DamageCategory.MINOR: (0.0, 2.5),
            DamageCategory.MODERATE: (2.5, 5.0),
            DamageCategory.SEVERE: (5.0, 7.5),
            DamageCategory.CRITICAL: (7.5, 10.0),
        }[self]

    @classmethod
    def from_score(cls, score: float) -> "DamageCategory":
        """
        Determine the damage category from a severity score.

        Args:
            score: Severity score between 0.0 and 10.0

        Returns:
            The corresponding DamageCategory

        Raises:
            ValueError: If score is outside [0, 10] range
        """
        if not 0.0 <= score <= 10.0:
            raise ValueError(f"Severity score must be 0-10, got {score}")

        for category in cls:
            low, high = category.score_range
            if low <= score < high:
                return category

        return cls.CRITICAL  # score == 10.0
