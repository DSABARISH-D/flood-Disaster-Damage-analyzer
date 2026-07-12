"""
Classifier Interface (Port)
=============================
Abstract contract that any flood classifier must implement.

This is the DEPENDENCY INVERSION principle in action:
  - Use cases depend on THIS interface (abstraction)
  - ResNetClassifier implements THIS interface (concrete)
  - The use case never imports ResNet, PyTorch, or TensorFlow

To add a new classifier (e.g., EfficientNet, ViT):
  1. Create a new class that implements IClassifier
  2. Register it in the dependency injection container
  3. No existing code needs to change
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class ClassificationResult:
    """
    Standard output from any classifier implementation.

    Attributes:
        label: Predicted class ("flood" or "no_flood")
        confidence: Prediction confidence (0.0 – 1.0)
        is_flood: Boolean shortcut for flood detection
        probabilities: Per-class probability distribution
        metadata: Optional model-specific extra data
    """

    def __init__(
        self,
        label: str,
        confidence: float,
        is_flood: bool,
        probabilities: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.label = label
        self.confidence = confidence
        self.is_flood = is_flood
        self.probabilities = probabilities or {}
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "is_flood": self.is_flood,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "metadata": self.metadata,
        }


class IClassifier(ABC):
    """
    Abstract interface for image classification.

    Any classifier (ResNet50, EfficientNet, ViT, etc.)
    must implement this interface to be used in the pipeline.
    """

    @abstractmethod
    def classify(self, image: Image.Image) -> ClassificationResult:
        """
        Classify an image as flood or no-flood.

        Args:
            image: PIL Image in RGB format.

        Returns:
            ClassificationResult with label, confidence, and probabilities.

        Raises:
            RuntimeError: If the model fails to produce a prediction.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model weights are loaded and ready for inference."""
        ...
