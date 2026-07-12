"""
Application Settings
=====================
Centralized configuration loaded from environment variables.
Uses Pydantic BaseSettings for validation and type coercion.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration — all values loaded from .env file
    or environment variables.
    """

    # ── Application ────────────────────────────────────
    APP_NAME: str = "Flood Damage Assessment API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Supabase ───────────────────────────────────────
    SUPABASE_URL: str = Field(default="", description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(default="", description="Supabase anon/public key")
    SUPABASE_SERVICE_KEY: str = Field(default="", description="Supabase service role key (server-only)")

    # ── Storage Buckets ────────────────────────────────
    BUCKET_ORIGINAL_IMAGES: str = "original-images"
    BUCKET_RESULT_IMAGES: str = "result-images"
    BUCKET_REPORTS: str = "reports"

    # ── CORS ───────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── AI Models ──────────────────────────────────────
    DEVICE: str = "cpu"  # "cpu" or "cuda"

    # ResNet50
    RESNET_WEIGHTS_PATH: str = ""  # Path to fine-tuned .pth weights (empty = demo mode)
    RESNET_CONFIDENCE_THRESHOLD: float = 0.5

    # SegFormer
    SEGFORMER_MODEL_ID: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    SEGFORMER_WATER_CLASS_IDS: list[int] = [21, 60, 26]

    # YOLOv8
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.45

    # ── Preprocessing ──────────────────────────────────
    PREPROCESS_TARGET_SIZE: int = 640
    IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
    DENOISE_STRENGTH: int = 10

    # ── Damage Assessment Weights ──────────────────────
    DAMAGE_WEIGHT_FLOOD_AREA: float = 0.40
    DAMAGE_WEIGHT_STRUCTURES: float = 0.30
    DAMAGE_WEIGHT_OBJECT_TYPES: float = 0.30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance
settings = Settings()
