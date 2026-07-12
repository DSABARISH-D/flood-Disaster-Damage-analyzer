"""
Preprocessor Interface (Port)
===============================
Abstract contract for image preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
import torch

from backend.domain.value_objects.image_metadata import ImageMetadata
from backend.domain.value_objects.geo_coordinate import GeoCoordinate


class PreprocessedImage:
    """
    Standard output from any preprocessor implementation.

    Contains all the different representations of the image
    needed by the various AI models in the pipeline.
    """

    def __init__(
        self,
        original_bgr: np.ndarray,
        display_rgb: np.ndarray,
        pil_image: Image.Image,
        tensor: torch.Tensor,
        metadata: ImageMetadata,
        gps: GeoCoordinate | None = None,
    ):
        self.original_bgr = original_bgr      # BGR for OpenCV operations
        self.display_rgb = display_rgb          # RGB for display / matplotlib
        self.pil_image = pil_image              # PIL for HuggingFace models
        self.tensor = tensor                    # [1,3,224,224] for ResNet50
        self.metadata = metadata                # Image metadata
        self.gps = gps                          # Extracted GPS coordinates


class IPreprocessor(ABC):
    """
    Abstract interface for image preprocessing.

    Any preprocessor (OpenCV-based, albumentations-based, etc.)
    must implement this interface.
    """

    @abstractmethod
    def preprocess(self, image_bytes: bytes, filename: str = "") -> PreprocessedImage:
        """
        Preprocess raw image bytes into model-ready formats.

        Args:
            image_bytes: Raw file bytes from upload.
            filename: Original filename (for metadata).

        Returns:
            PreprocessedImage with all necessary representations.
        """
        ...
