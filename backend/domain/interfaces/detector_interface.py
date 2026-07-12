"""
Detector Interface (Port)
===========================
Abstract contract for object detection models.
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from backend.domain.entities.detection import Detection


class DetectionResult:
    """
    Standard output from any detector implementation.

    Attributes:
        detections: List of Detection entities
        annotated_image: Image with drawn bounding boxes (numpy RGB)
        total_objects: Total number of detected objects
        objects_in_flood: Number of objects in the flood zone
        by_class: Count of objects grouped by class name
    """

    def __init__(
        self,
        detections: list[Detection],
        annotated_image: np.ndarray | None = None,
    ):
        self.detections = detections
        self.annotated_image = annotated_image

    @property
    def total_objects(self) -> int:
        return len(self.detections)

    @property
    def objects_in_flood(self) -> int:
        return sum(1 for d in self.detections if d.in_flood_zone)

    @property
    def by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            name = d.object_class.value
            counts[name] = counts.get(name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "total_objects": self.total_objects,
            "objects_in_flood": self.objects_in_flood,
            "by_class": self.by_class,
        }


class IDetector(ABC):
    """
    Abstract interface for object detection.

    Any detector (YOLOv8, Faster R-CNN, DETR, etc.)
    must implement this interface.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        flood_mask: np.ndarray | None = None,
    ) -> DetectionResult:
        """
        Detect objects in an image, optionally checking flood zone overlap.

        Args:
            image: RGB numpy array (H, W, 3).
            flood_mask: Optional binary mask (H, W) with 255 = flood.

        Returns:
            DetectionResult with list of Detection entities.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        ...
