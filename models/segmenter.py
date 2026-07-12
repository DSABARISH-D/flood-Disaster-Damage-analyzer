"""
SegFormer Flood Segmentation Module
=====================================
Performs pixel-level semantic segmentation to identify
flood / water regions in an image using SegFormer from HuggingFace.
"""

import torch
import numpy as np
import cv2
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class FloodSegmenter:
    """
    SegFormer-based semantic segmentation for flood area mapping.

    Uses a pretrained model from HuggingFace (default: ADE20K-finetuned)
    and extracts water/flood classes to create a binary flood mask.
    """

    def __init__(self, model_id=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or config.SEGFORMER_MODEL_ID
        self.water_class_ids = config.SEGFORMER_WATER_CLASS_IDS
        self.input_size = config.SEGFORMER_INPUT_SIZE
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SegFormer model and processor from HuggingFace."""
        try:
            # Prefer explicit SegFormer classes when available
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except Exception:
            # Fall back to generic Auto classes for broader compatibility
            from transformers import (
                AutoImageProcessor as SegformerImageProcessor,
                AutoModelForSemanticSegmentation as SegformerForSemanticSegmentation,
            )

        print(f"[Segmenter] Loading SegFormer model: {self.model_id}")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"[Segmenter] Model loaded on {self.device}")
        except Exception as exc:
            # Don't raise raw exceptions during import; store None and log helpful message
            print(f"[Segmenter][ERROR] Failed to load SegFormer model '{self.model_id}': {exc}")
            print("[Segmenter][HINT] Check network access, HF token limits, and that 'transformers' is installed.")
            self.model = None
            self.processor = None

    @torch.no_grad()
    def segment(self, pil_image: Image.Image) -> dict:
        """
        Perform semantic segmentation on an image.

        Args:
            pil_image: PIL Image in RGB format.

        Returns:
            dict: {
                "flood_mask": np.ndarray (H, W) binary mask (0 or 255),
                "full_segmap": np.ndarray (H, W) full segmentation map,
                "flood_area_ratio": float (0-1),
                "flood_overlay": np.ndarray (H, W, 3) colored overlay,
                "num_labels": int,
            }
        """
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape: (1, num_classes, H/4, W/4)

        # Upsample to original image size
        original_size = pil_image.size[::-1]  # (H, W)
        upsampled = torch.nn.functional.interpolate(
            logits, size=original_size, mode="bilinear", align_corners=False
        )

        # Get per-pixel class predictions
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Create binary flood mask from water classes
        flood_mask = np.zeros_like(seg_map, dtype=np.uint8)
        for class_id in self.water_class_ids:
            flood_mask[seg_map == class_id] = 255

        # Apply morphological operations to clean up the mask
        flood_mask = self._clean_mask(flood_mask)

        # Calculate flood area ratio
        total_pixels = flood_mask.shape[0] * flood_mask.shape[1]
        flood_pixels = np.count_nonzero(flood_mask)
        flood_area_ratio = flood_pixels / total_pixels if total_pixels > 0 else 0.0

        # Create colored overlay
        flood_overlay = self._create_overlay(flood_mask, original_size)

        return {
            "flood_mask": flood_mask,
            "full_segmap": seg_map,
            "flood_area_ratio": flood_area_ratio,
            "flood_overlay": flood_overlay,
            "num_labels": int(self.model.config.num_labels),
        }

    def create_overlay_on_image(self, image: np.ndarray, flood_mask: np.ndarray, alpha=None) -> np.ndarray:
        """
        Overlay the flood mask on the original image.

        Args:
            image: RGB numpy array (H, W, 3)
            flood_mask: Binary mask (H, W), 255 = flood
            alpha: Transparency for flood overlay

        Returns:
            np.ndarray: Image with blue flood overlay
        """
        alpha = alpha or config.VIS_FLOOD_ALPHA
        overlay = image.copy()

        # Resize mask to match image if needed
        if flood_mask.shape[:2] != image.shape[:2]:
            flood_mask = cv2.resize(flood_mask, (image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Apply blue overlay on flood regions
        flood_color = np.array(config.VIS_FLOOD_COLOR, dtype=np.uint8)
        mask_bool = flood_mask > 0
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + flood_color * alpha
        ).astype(np.uint8)

        return overlay

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological ops to remove noise from segmentation mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _create_overlay(self, flood_mask: np.ndarray, size: tuple) -> np.ndarray:
        """Create a colored RGBA overlay image from the flood mask."""
        h, w = flood_mask.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[flood_mask > 0] = config.VIS_FLOOD_COLOR
        return overlay
