"""
Image Metadata Value Object
=============================
Captures metadata about an uploaded image
(dimensions, format, size) without holding the image data itself.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageMetadata:
    """
    Immutable metadata for an uploaded image.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels (3 for RGB, 4 for RGBA)
        format: File format string (e.g., "JPEG", "PNG")
        size_bytes: File size in bytes
        filename: Original filename from upload
    """

    width: int
    height: int
    channels: int = 3
    format: str = "JPEG"
    size_bytes: int = 0
    filename: str = ""

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.width}x{self.height}")

    @property
    def resolution(self) -> str:
        """Human-readable resolution string."""
        return f"{self.width}×{self.height}"

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height aspect ratio."""
        return self.width / self.height

    @property
    def megapixels(self) -> float:
        """Total megapixels."""
        return (self.width * self.height) / 1_000_000

    @property
    def size_human(self) -> str:
        """Human-readable file size (e.g., '2.4 MB')."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 ** 2:
            return f"{self.size_bytes / 1024:.1f} KB"
        else:
            return f"{self.size_bytes / (1024 ** 2):.1f} MB"

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "size_human": self.size_human,
            "resolution": self.resolution,
            "megapixels": round(self.megapixels, 2),
            "filename": self.filename,
        }
