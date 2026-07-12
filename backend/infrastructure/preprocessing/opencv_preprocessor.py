"""
OpenCV Preprocessor — Infrastructure Implementation
=====================================================
Implements IPreprocessor using OpenCV for image manipulation.

Pipeline: Load → Resize → Denoise → Normalize → Convert formats
"""

import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.domain.interfaces.preprocessor_interface import IPreprocessor, PreprocessedImage
from backend.domain.value_objects.image_metadata import ImageMetadata
from backend.domain.value_objects.geo_coordinate import GeoCoordinate
from backend.config.settings import settings


class OpenCVPreprocessor(IPreprocessor):
    """
    OpenCV-based image preprocessor implementing IPreprocessor.

    Produces multiple representations of the input image:
      - BGR numpy array (for OpenCV operations)
      - RGB numpy array (for display)
      - PIL Image (for HuggingFace models)
      - Normalized PyTorch tensor (for ResNet50)
    """

    def __init__(self):
        self.target_size = settings.PREPROCESS_TARGET_SIZE
        self.denoise_strength = settings.DENOISE_STRENGTH

        self._tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD),
        ])

    def preprocess(self, image_bytes: bytes, filename: str = "") -> PreprocessedImage:
        """Convert raw bytes into all necessary image representations."""
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from bytes")

        # Build metadata
        h, w, c = bgr.shape
        metadata = ImageMetadata(
            width=w, height=h, channels=c,
            format=self._guess_format(filename),
            size_bytes=len(image_bytes),
            filename=filename,
        )

        # Resize
        resized_bgr = self._resize(bgr)

        # Denoise
        denoised = cv2.bilateralFilter(
            resized_bgr, d=9,
            sigmaColor=self.denoise_strength * 7.5,
            sigmaSpace=self.denoise_strength * 7.5,
        )

        # Convert to RGB
        display_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

        # PIL Image
        pil_image = Image.fromarray(display_rgb)

        # Tensor for ResNet50
        tensor = self._tensor_transform(pil_image).unsqueeze(0)

        # Extract GPS
        gps = self._extract_gps(image_bytes)

        return PreprocessedImage(
            original_bgr=denoised,
            display_rgb=display_rgb,
            pil_image=pil_image,
            tensor=tensor,
            metadata=metadata,
            gps=gps,
        )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize while maintaining aspect ratio, padding to square."""
        h, w = image.shape[:2]
        target = self.target_size
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    @staticmethod
    def _guess_format(filename: str) -> str:
        """Guess image format from filename extension."""
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "JPEG"
        return {"JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "TIFF": "TIFF", "BMP": "BMP"}.get(ext, "JPEG")

    @staticmethod
    def _extract_gps(image_bytes: bytes) -> GeoCoordinate | None:
        """Attempt to extract GPS coordinates from EXIF data."""
        try:
            import exifread
            tags = exifread.process_file(io.BytesIO(image_bytes), details=False)

            lat_tag = tags.get("GPS GPSLatitude")
            lat_ref = tags.get("GPS GPSLatitudeRef")
            lon_tag = tags.get("GPS GPSLongitude")
            lon_ref = tags.get("GPS GPSLongitudeRef")

            if not all([lat_tag, lat_ref, lon_tag, lon_ref]):
                return None

            def to_decimal(tag):
                vals = tag.values
                d = float(vals[0].num) / float(vals[0].den)
                m = float(vals[1].num) / float(vals[1].den)
                s = float(vals[2].num) / float(vals[2].den)
                return d + m / 60.0 + s / 3600.0

            lat = to_decimal(lat_tag) * (-1 if str(lat_ref) == "S" else 1)
            lon = to_decimal(lon_tag) * (-1 if str(lon_ref) == "W" else 1)

            return GeoCoordinate(lat=lat, lon=lon)
        except Exception:
            return None
