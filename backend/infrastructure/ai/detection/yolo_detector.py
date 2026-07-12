"""
YOLOv8 Detector — Infrastructure Implementation
==================================================
Implements IDetector using Ultralytics YOLOv8 model.

Detects objects from COCO classes, filters to disaster-relevant
categories, and checks overlap with the flood mask.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from backend.domain.interfaces.detector_interface import IDetector, DetectionResult
from backend.domain.entities.detection import Detection
from backend.domain.enums.object_class import ObjectClass
from backend.domain.value_objects.bounding_box import BoundingBox


class YOLODetector(IDetector):
    """
    YOLOv8-based object detector implementing IDetector.

    Filters COCO detections to disaster-relevant classes and
    tags objects with flood zone overlap status.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        flood_overlap_threshold: float = 0.3,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.flood_overlap_threshold = flood_overlap_threshold
        self._model = None
        self._target_ids = ObjectClass.target_coco_ids()
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load YOLOv8 model from Ultralytics."""
        self._model = YOLO(model_name)

    def detect(
        self,
        image: np.ndarray,
        flood_mask: np.ndarray | None = None,
    ) -> DetectionResult:
        """Detect objects, filter to target classes, check flood overlap."""
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []

        if results and len(results) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id not in self._target_ids:
                    continue

                obj_class = ObjectClass.from_coco_id(class_id)
                if obj_class is None:
                    continue

                confidence = float(boxes.conf[i].item())
                coords = boxes.xyxy[i].cpu().numpy().tolist()
                bbox = BoundingBox.from_xyxy(coords)

                # Check flood zone overlap
                in_flood = False
                if flood_mask is not None:
                    in_flood = self._check_flood_overlap(
                        flood_mask, bbox.x1, bbox.y1, bbox.x2, bbox.y2
                    )

                detections.append(Detection(
                    object_class=obj_class,
                    confidence=confidence,
                    bbox=bbox,
                    in_flood_zone=in_flood,
                ))

        # Draw annotated image
        annotated = self._draw_detections(image.copy(), detections)

        return DetectionResult(detections=detections, annotated_image=annotated)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _check_flood_overlap(
        self, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> bool:
        """Check if a bounding box overlaps with the flood mask."""
        h, w = mask.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        if x2c <= x1c or y2c <= y1c:
            return False

        roi = mask[y1c:y2c, x1c:x2c]
        flood_px = np.count_nonzero(roi)
        total_px = roi.size

        return (flood_px / total_px) >= self.flood_overlap_threshold if total_px > 0 else False

    @staticmethod
    def _draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox.as_xyxy
            color = (255, 50, 50) if det.in_flood_zone else (50, 255, 50)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{det.object_class.value} {det.confidence:.0%}"
            if det.in_flood_zone:
                label += " FLOOD"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return image
