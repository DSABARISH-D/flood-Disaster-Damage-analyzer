"""
YOLOv8 Object Detection Module
================================
Detects buildings, vehicles, people, and other infrastructure
in flood-affected regions using YOLOv8 from Ultralytics.
"""

import numpy as np
import cv2
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ObjectDetector:
    """
    YOLOv8-based object detector for disaster damage assessment.

    Uses COCO-pretrained weights to detect vehicles, people, boats,
    and other relevant objects. Filters results to only disaster-relevant classes.
    """

    def __init__(self, model_name=None, device=None):
        self.device = device or ("cuda:0" if self._cuda_available() else "cpu")
        self.model_name = model_name or config.YOLO_MODEL_NAME
        self.confidence_threshold = config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.target_classes = config.YOLO_TARGET_CLASSES
        self.model = None
        self._load_model()

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self):
        """Load YOLOv8 model from Ultralytics."""
        try:
            from ultralytics import YOLO
        except Exception as exc:
            print(f"[Detector][ERROR] Failed to import 'ultralytics': {exc}")
            self.model = None
            return

        print(f"[Detector] Loading YOLOv8 model: {self.model_name}")
        try:
            self.model = YOLO(self.model_name)
            print(f"[Detector] Model loaded on {self.device}")
        except Exception as exc:
            print(f"[Detector][ERROR] Failed to load YOLO model '{self.model_name}': {exc}")
            self.model = None

    def detect(self, image: np.ndarray, flood_mask: np.ndarray = None) -> dict:
        """
        Run object detection on an image.

        Args:
            image: BGR or RGB numpy array (H, W, 3)
            flood_mask: Optional binary mask (H, W) with 255 = flood region.
                        Used to tag objects as being in the flood zone.

        Returns:
            dict: {
                "detections": list of detection dicts,
                "annotated_image": np.ndarray with drawn bboxes,
                "summary": {
                    "total_objects": int,
                    "objects_in_flood": int,
                    "by_class": {"car": 3, "person": 1, ...}
                }
            }
        """
        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=config.YOLO_INPUT_SIZE,
            device=self.device,
            verbose=False,
        )

        detections = []
        class_counts = {}
        objects_in_flood = 0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Filter to only target classes
                if class_id not in self.target_classes:
                    continue

                class_name = self.target_classes[class_id]
                confidence = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Check if object is in flood zone
                in_flood_zone = False
                if flood_mask is not None:
                    in_flood_zone = self._check_flood_overlap(
                        flood_mask, x1, y1, x2, y2
                    )
                    if in_flood_zone:
                        objects_in_flood += 1

                detection = {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    "area_px": (x2 - x1) * (y2 - y1),
                    "in_flood_zone": in_flood_zone,
                }
                detections.append(detection)

                # Count by class
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Create annotated image
        annotated_image = self._draw_detections(image.copy(), detections)

        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "summary": {
                "total_objects": len(detections),
                "objects_in_flood": objects_in_flood,
                "by_class": class_counts,
            },
        }

    def _check_flood_overlap(
        self, flood_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int,
        overlap_threshold: float = 0.3
    ) -> bool:
        """
        Check if a bounding box overlaps with the flood mask.

        Returns True if > overlap_threshold of the bbox area is flooded.
        """
        # Resize mask if needed
        mask = flood_mask
        if mask.shape[:2] != (flood_mask.shape[0], flood_mask.shape[1]):
            return False

        # Clip coordinates to mask bounds
        h, w = mask.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        if x2c <= x1c or y2c <= y1c:
            return False

        roi = mask[y1c:y2c, x1c:x2c]
        flood_pixels = np.count_nonzero(roi)
        total_pixels = roi.size

        if total_pixels == 0:
            return False

        return (flood_pixels / total_pixels) >= overlap_threshold

    def _draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class"]
            confidence = det["confidence"]
            in_flood = det["in_flood_zone"]

            # Get color
            color = config.VIS_BBOX_COLORS.get(class_name, config.VIS_BBOX_COLORS["default"])
            if in_flood:
                # Red border for objects in flood zone
                color = (255, 50, 50)

            thickness = config.VIS_BBOX_THICKNESS
            font_scale = config.VIS_FONT_SCALE

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"{class_name} {confidence:.0%}"
            if in_flood:
                label += " [FLOOD]"

            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                image, (x1, y1 - label_h - 10), (x1 + label_w + 4, y1), color, -1
            )
            cv2.putText(
                image, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return image
