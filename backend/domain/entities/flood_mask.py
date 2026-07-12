"""
Flood Mask Entity
==================
Represents the output of semantic segmentation —
a pixel-level map of flood/water regions in the image.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FloodMask:
    """
    Represents the segmentation output — where water/flood exists in the image.

    NOTE: The actual numpy mask array is NOT stored here to keep the
    domain layer free of numpy dependencies. Instead, it holds metadata
    and the mask is passed separately through the pipeline.

    Attributes:
        flood_area_ratio: Fraction of image covered by flood (0.0 – 1.0)
        flood_pixel_count: Number of pixels classified as flood
        total_pixel_count: Total pixels in the image
        mask_image_url: URL of the saved mask overlay image
        num_segments: Number of distinct flood regions found
    """

    flood_area_ratio: float
    flood_pixel_count: int
    total_pixel_count: int
    mask_image_url: str = ""
    num_segments: int = 0

    def __post_init__(self):
        if not 0.0 <= self.flood_area_ratio <= 1.0:
            raise ValueError(f"Flood area ratio must be 0-1, got {self.flood_area_ratio}")

    @property
    def flood_area_percentage(self) -> float:
        """Flood area as a human-readable percentage."""
        return round(self.flood_area_ratio * 100, 2)

    @property
    def non_flood_ratio(self) -> float:
        """Fraction of image NOT covered by flood."""
        return 1.0 - self.flood_area_ratio

    @property
    def is_significant(self) -> bool:
        """Whether flood coverage is significant (> 5% of image)."""
        return self.flood_area_ratio > 0.05

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "flood_area_ratio": round(self.flood_area_ratio, 4),
            "flood_area_percentage": self.flood_area_percentage,
            "flood_pixel_count": self.flood_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "mask_image_url": self.mask_image_url,
            "num_segments": self.num_segments,
        }
