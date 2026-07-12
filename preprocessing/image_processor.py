"""
OpenCV Image Preprocessing Module
==================================
Handles image loading, resizing, normalization, and noise removal
before feeding into the deep learning models.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ImageProcessor:
    """Preprocesses uploaded images for the ML pipeline."""

    def __init__(
        self,
        target_size=None,
        normalize_mean=None,
        normalize_std=None,
        denoise_strength=None,
    ):
        self.target_size = target_size or tuple(config.PREPROCESS_TARGET_SIZE)
        self.normalize_mean = normalize_mean or config.PREPROCESS_NORMALIZE_MEAN
        self.normalize_std = normalize_std or config.PREPROCESS_NORMALIZE_STD
        self.denoise_strength = denoise_strength or config.PREPROCESS_DENOISE_STRENGTH

        # PyTorch transform pipeline for ResNet50
        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])

    # ── Public API ─────────────────────────────────────────

    def load_image(self, source) -> np.ndarray:
        """
        Load an image from a file path, file-like object, or PIL Image.
        Returns a BGR numpy array (OpenCV format).
        """
        if isinstance(source, np.ndarray):
            return source.copy()
        elif isinstance(source, Image.Image):
            rgb = np.array(source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(source, str):
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return img
        else:
            # File-like object (e.g., Streamlit UploadedFile)
            file_bytes = np.frombuffer(source.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image from uploaded file.")
            # Reset file pointer if possible
            if hasattr(source, "seek"):
                source.seek(0)
            return img

    def preprocess(self, image: np.ndarray) -> dict:
        """
        Full preprocessing pipeline.

        Returns:
            dict with keys:
                - "original"   : original BGR image (resized)
                - "display"    : RGB image for display
                - "denoised"   : denoised BGR image
                - "tensor"     : normalized PyTorch tensor [1, 3, 224, 224]
                - "pil"        : PIL Image (RGB)
                - "shape"      : (height, width) of resized image
        """
        # Step 1: Resize
        resized = self._resize(image)

        # Step 2: Denoise
        denoised = self._denoise(resized)

        # Step 3: Convert to RGB for display & PIL
        display_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_rgb)

        # Step 4: Create normalized tensor for ResNet50
        tensor = self.tensor_transform(pil_image).unsqueeze(0)

        return {
            "original": resized,
            "display": display_rgb,
            "denoised": denoised,
            "tensor": tensor,
            "pil": pil_image,
            "shape": resized.shape[:2],
        }

    def preprocess_for_segformer(self, image: np.ndarray) -> np.ndarray:
        """Resize image for SegFormer input (512x512)."""
        h, w = config.SEGFORMER_INPUT_SIZE
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    def preprocess_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """Resize image for YOLOv8 input (640x640)."""
        size = config.YOLO_INPUT_SIZE
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    # ── Private Helpers ────────────────────────────────────

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize while maintaining aspect ratio, then pad/crop to target."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to exact target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise removal."""
        return cv2.bilateralFilter(
            image,
            d=9,
            sigmaColor=self.denoise_strength * 7.5,
            sigmaSpace=self.denoise_strength * 7.5,
        )

    @staticmethod
    def extract_exif_gps(source) -> dict | None:
        """
        Attempt to extract GPS coordinates from image EXIF data.
        Returns {"lat": float, "lon": float} or None.
        """
        try:
            import exifread

            if isinstance(source, str):
                with open(source, "rb") as f:
                    tags = exifread.process_file(f, details=False)
            elif hasattr(source, "read"):
                tags = exifread.process_file(source, details=False)
                if hasattr(source, "seek"):
                    source.seek(0)
            else:
                return None

            lat_tag = tags.get("GPS GPSLatitude")
            lat_ref = tags.get("GPS GPSLatitudeRef")
            lon_tag = tags.get("GPS GPSLongitude")
            lon_ref = tags.get("GPS GPSLongitudeRef")

            if not all([lat_tag, lat_ref, lon_tag, lon_ref]):
                return None

            def _to_decimal(tag):
                vals = tag.values
                d = float(vals[0].num) / float(vals[0].den)
                m = float(vals[1].num) / float(vals[1].den)
                s = float(vals[2].num) / float(vals[2].den)
                return d + m / 60.0 + s / 3600.0

            lat = _to_decimal(lat_tag)
            lon = _to_decimal(lon_tag)
            if str(lat_ref) == "S":
                lat = -lat
            if str(lon_ref) == "W":
                lon = -lon

            return {"lat": lat, "lon": lon}
        except Exception:
            return None
