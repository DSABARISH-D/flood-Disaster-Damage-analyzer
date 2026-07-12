"""
ResNet50 Flood Classifier
==========================
Binary classifier that determines whether an image contains
a flood event or not. Uses pretrained ResNet50 as backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class FloodClassifier:
    """
    ResNet50-based binary classifier for flood detection.

    Modes:
      - Pretrained demo mode (no custom weights): uses ImageNet features
        with a randomly initialized head — fine-tuning required for production.
      - Custom weights mode: loads user-provided .pth weights.
    """

    def __init__(self, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = config.CLASSIFIER_LABELS
        self.threshold = config.CLASSIFIER_CONFIDENCE_THRESHOLD
        self.model = self._build_model(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, weights_path=None):
        """Build ResNet50 with modified final layer for binary classification if custom weights exist."""
        # Load pretrained ResNet50 backbone
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            # Replace final fully-connected layer for flood detection
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, config.CLASSIFIER_NUM_CLASSES),
            )
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.is_fine_tuned = True
            print(f"[Classifier] Loaded custom weights from {weights_path}")
        else:
            self.is_fine_tuned = False
            print("[Classifier] Using pure pretrained ResNet50 ImageNet (no flood fine-tuning).")

        return model

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        """
        Classify an image tensor.

        Args:
            tensor: Preprocessed tensor of shape [1, 3, 224, 224]

        Returns:
            dict: {
                "label": "flood" | "no_flood",
                "confidence": float (0-1),
                "probabilities": {"flood": float, "no_flood": float},
                "is_flood": bool
            }
        """
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_confidence = float(probs[pred_idx])

        if self.is_fine_tuned:
            pred_label = self.labels[pred_idx]
            return {
                "label": pred_label,
                "confidence": pred_confidence,
                "is_flood": pred_label == "flood" and pred_confidence >= self.threshold,
                "imagenet_class": None
            }
        else:
            # Return pure ImageNet class, do not claim flood detection
            categories = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
            imagenet_class = categories[pred_idx]
            return {
                "label": "Unable to determine with current pretrained model",
                "confidence": pred_confidence,
                "is_flood": None,
                "imagenet_class": imagenet_class
            }

    @staticmethod
    def _rgb_to_hsv_numpy(rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image (0-255) to HSV with H in [0,180], S,V in [0,255]."""
        import cv2
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
