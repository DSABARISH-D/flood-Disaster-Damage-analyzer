"""
Bounding Box Value Object
==========================
Represents a rectangular region in an image defined by
its top-left and bottom-right corners.

Immutable. Two BoundingBoxes with the same coordinates are equal.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box for object detection results.

    Coordinate system: (0,0) is top-left of image.
      x1, y1 = top-left corner
      x2, y2 = bottom-right corner

    Invariant: x1 <= x2 and y1 <= y2.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        if self.x1 > self.x2:
            raise ValueError(f"x1 ({self.x1}) must be <= x2 ({self.x2})")
        if self.y1 > self.y2:
            raise ValueError(f"y1 ({self.y1}) must be <= y2 ({self.y2})")

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def as_xyxy(self) -> tuple[int, int, int, int]:
        """Return coordinates in (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def as_xywh(self) -> tuple[int, int, int, int]:
        """Return in (x, y, width, height) format (top-left origin)."""
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box.

        Returns:
            float: IoU value between 0.0 (no overlap) and 1.0 (identical).
        """
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width": self.width, "height": self.height,
            "area": self.area,
            "center": list(self.center),
        }

    @classmethod
    def from_xyxy(cls, coords: list[int | float]) -> BoundingBox:
        """Create from [x1, y1, x2, y2] list."""
        return cls(x1=int(coords[0]), y1=int(coords[1]),
                   x2=int(coords[2]), y2=int(coords[3]))
