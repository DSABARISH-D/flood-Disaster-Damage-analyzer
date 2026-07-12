"""
SegFormer Segmenter — Infrastructure Implementation
=====================================================
Implements ISegmenter using HuggingFace SegFormer model.

Extracts water/flood classes from the ADE20K segmentation map
and produces a binary flood mask with morphological cleanup.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from backend.domain.interfaces.segmenter_interface import ISegmenter, SegmentationResult


class SegFormerSegmenter(ISegmenter):
    """
    SegFormer-based semantic segmenter implementing ISegmenter.

    Uses pretrained ADE20K model and extracts water/river/sea classes
    to create a binary flood mask.
    """

    def __init__(
        self,
        model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu",
        water_class_ids: list[int] | None = None,
    ):
        self.device = device
        self.model_id = model_id
        self.water_class_ids = water_class_ids or [21, 60, 26]  # water, river, sea
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        """Load SegFormer model and processor from HuggingFace."""
        self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image) -> SegmentationResult:
        """Perform semantic segmentation to identify flood regions."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        logits = outputs.logits

        # Upsample to original size
        original_size = image.size[::-1]  # (H, W)
        upsampled = torch.nn.functional.interpolate(
            logits, size=original_size, mode="bilinear", align_corners=False
        )

        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Build binary flood mask from water classes
        flood_mask = np.zeros_like(seg_map, dtype=np.uint8)
        for class_id in self.water_class_ids:
            flood_mask[seg_map == class_id] = 255

        # Morphological cleanup
        flood_mask = self._clean_mask(flood_mask)

        # Compute metrics
        total_pixels = flood_mask.shape[0] * flood_mask.shape[1]
        flood_pixels = int(np.count_nonzero(flood_mask))
        area_ratio = flood_pixels / total_pixels if total_pixels > 0 else 0.0

        return SegmentationResult(
            flood_mask=flood_mask,
            flood_area_ratio=area_ratio,
            flood_pixel_count=flood_pixels,
            total_pixel_count=total_pixels,
            metadata={"model_id": self.model_id, "water_class_ids": self.water_class_ids},
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological ops to remove noise from segmentation mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
