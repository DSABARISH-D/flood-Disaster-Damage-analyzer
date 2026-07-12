"""
Segmenter Interface (Port)
============================
Abstract contract for flood segmentation models.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image


class SegmentationResult:
    """
    Standard output from any segmenter implementation.

    Attributes:
        flood_mask: Binary numpy mask (H, W) — 255 = flood, 0 = no flood
        flood_area_ratio: Fraction of image covered by flood (0.0 – 1.0)
        flood_pixel_count: Number of pixels classified as flood
        total_pixel_count: Total pixels in the mask
        metadata: Optional model-specific extra data
    """

    def __init__(
        self,
        flood_mask: np.ndarray,
        flood_area_ratio: float,
        flood_pixel_count: int,
        total_pixel_count: int,
        metadata: dict[str, Any] | None = None,
    ):
        self.flood_mask = flood_mask
        self.flood_area_ratio = flood_area_ratio
        self.flood_pixel_count = flood_pixel_count
        self.total_pixel_count = total_pixel_count
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Serialize (excluding the numpy mask)."""
        return {
            "flood_area_ratio": round(self.flood_area_ratio, 4),
            "flood_area_percentage": round(self.flood_area_ratio * 100, 2),
            "flood_pixel_count": self.flood_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "metadata": self.metadata,
        }


class ISegmenter(ABC):
    """
    Abstract interface for flood segmentation.

    Any segmenter (SegFormer, U-Net, DeepLab, etc.)
    must implement this interface.
    """

    @abstractmethod
    def segment(self, image: Image.Image) -> SegmentationResult:
        """
        Perform semantic segmentation to identify flood regions.

        Args:
            image: PIL Image in RGB format.

        Returns:
            SegmentationResult with binary flood mask and area metrics.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        ...
