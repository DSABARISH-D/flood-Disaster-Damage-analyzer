"""
ResNet50 Classifier — Infrastructure Implementation
======================================================
Implements IClassifier using PyTorch ResNet50 backbone.

Supports two modes:
  1. DEMO MODE: Pretrained ImageNet weights + color heuristic (no fine-tuning needed)
  2. PRODUCTION MODE: Fine-tuned weights loaded from a .pth file
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from backend.domain.interfaces.classifier_interface import IClassifier, ClassificationResult


class ResNetClassifier(IClassifier):
    """
    ResNet50-based binary flood classifier.

    Implements the IClassifier interface from the domain layer.
    """

    LABELS = ["no_flood", "flood"]

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model: nn.Module | None = None
        self._weights_path = weights_path

        # Image transform pipeline
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._load_model()

    def _load_model(self):
        """Build and load the ResNet50 model."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace final FC for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

        # Load fine-tuned weights if available
        if self._weights_path and os.path.exists(self._weights_path):
            state_dict = torch.load(self._weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self._is_fine_tuned = True
        else:
            self._is_fine_tuned = False

        model.to(self.device)
        model.eval()
        self._model = model

    def classify(self, image: Image.Image) -> ClassificationResult:
        """
        Classify an image as flood or no-flood.

        Uses model output + color heuristic in demo mode.
        Uses only model output in production mode (fine-tuned weights).
        """
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        model_flood_prob = float(probs[1])

        # In demo mode, supplement with color heuristic
        if not self._is_fine_tuned:
            heuristic_prob = self._color_heuristic(image)
            combined_prob = 0.3 * model_flood_prob + 0.7 * heuristic_prob
        else:
            combined_prob = model_flood_prob

        is_flood = combined_prob >= self.confidence_threshold
        label = "flood" if is_flood else "no_flood"
        confidence = combined_prob if is_flood else (1.0 - combined_prob)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            is_flood=is_flood,
            probabilities={"flood": combined_prob, "no_flood": 1.0 - combined_prob},
            metadata={
                "model_flood_prob": float(model_flood_prob),
                "is_fine_tuned": self._is_fine_tuned,
            },
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def _color_heuristic(self, pil_image: Image.Image) -> float:
        """
        Analyze water/flood presence via HSV color distribution.

        Detects:
          - Blue water (H: 90-135, S: 30+)
          - Muddy brown water (H: 8-35, S: 40+)
        """
        img = np.array(pil_image.resize((224, 224)))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        blue_mask = (h >= 90) & (h <= 135) & (s >= 30) & (v >= 30)
        brown_mask = (h >= 8) & (h <= 35) & (s >= 40) & (v >= 40)

        water_ratio = (blue_mask.sum() + brown_mask.sum()) / (h.shape[0] * h.shape[1])
        return min(float(water_ratio) * 3.0, 1.0)
