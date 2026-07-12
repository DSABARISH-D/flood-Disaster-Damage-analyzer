"""
Severity Value Object
======================
Represents a damage severity score with automatic category derivation.
Immutable — once created, cannot be modified.

This is a VALUE OBJECT (not an Entity) because:
  - Two Severity objects with score=7.2 are considered EQUAL.
  - It has no identity (no ID field).
  - It describes a measurement, not a thing.
"""

from __future__ import annotations
from dataclasses import dataclass
from backend.domain.enums.damage_category import DamageCategory


@dataclass(frozen=True)
class Severity:
    """
    Immutable severity score with auto-derived category.

    Attributes:
        score: Numerical severity (0.0 – 10.0)
        category: Auto-derived DamageCategory
        emoji: Visual indicator from category

    Usage:
        sev = Severity(score=7.2)
        print(sev.category)  # DamageCategory.SEVERE
        print(sev.emoji)     # "🔴"
    """

    score: float

    def __post_init__(self):
        if not 0.0 <= self.score <= 10.0:
            raise ValueError(f"Severity score must be 0.0–10.0, got {self.score}")

    @property
    def category(self) -> DamageCategory:
        """Automatically derived damage category based on score."""
        return DamageCategory.from_score(self.score)

    @property
    def emoji(self) -> str:
        """Visual emoji indicator."""
        return self.category.emoji

    @property
    def color_hex(self) -> str:
        """CSS hex color for this severity level."""
        return self.category.color_hex

    @property
    def is_critical(self) -> bool:
        """Whether this severity is at critical level."""
        return self.category == DamageCategory.CRITICAL

    @property
    def is_actionable(self) -> bool:
        """Whether this severity level warrants immediate action (Severe or Critical)."""
        return self.category in (DamageCategory.SEVERE, DamageCategory.CRITICAL)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "score": round(self.score, 2),
            "category": self.category.value,
            "emoji": self.emoji,
            "color": self.color_hex,
        }
