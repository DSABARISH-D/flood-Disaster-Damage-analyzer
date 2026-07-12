"""
AI Module Configuration
=======================
Handles device selection and model paths.
"""
import torch

class AIConfig:
    # Use GPU if available, else fallback to CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Repos / Weights
    SEGFORMER_REPO = "nvidia/segformer-b0-finetuned-ade-512-512"
    YOLO_WEIGHTS = "yolov8n.pt"  # Will auto-download if missing
    
    # Thresholds
    CLASSIFICATION_THRESHOLD = 0.5  # Confidence required to declare "Flood"
    DETECTION_CONFIDENCE = 0.25     # Confidence required for YOLO boxes

config = AIConfig()
