"""
Dependency Injection Container
================================
Wires abstract interfaces to concrete implementations.

This is where the DEPENDENCY INVERSION principle comes together:
  - Use cases depend on IClassifier (abstraction)
  - This module provides ResNetClassifier (concrete)
  - FastAPI injects the concrete via Depends()

To swap an implementation (e.g., ResNet → EfficientNet):
  → Change ONE line here. Nothing else changes.
"""

from functools import lru_cache

from backend.config.settings import settings


# ── Lazy Singletons ───────────────────────────────────────
# Models are expensive to load; we cache them as singletons.

@lru_cache(maxsize=1)
def get_preprocessor():
    """Provide the IPreprocessor implementation."""
    from backend.infrastructure.preprocessing.opencv_preprocessor import OpenCVPreprocessor
    return OpenCVPreprocessor()


@lru_cache(maxsize=1)
def get_classifier():
    """Provide the IClassifier implementation."""
    from backend.infrastructure.ai.classification.resnet_classifier import ResNetClassifier
    return ResNetClassifier(
        weights_path=settings.RESNET_WEIGHTS_PATH or None,
        device=settings.DEVICE,
        confidence_threshold=settings.RESNET_CONFIDENCE_THRESHOLD,
    )


@lru_cache(maxsize=1)
def get_segmenter():
    """Provide the ISegmenter implementation."""
    from backend.infrastructure.ai.segmentation.segformer_segmenter import SegFormerSegmenter
    return SegFormerSegmenter(
        model_id=settings.SEGFORMER_MODEL_ID,
        device=settings.DEVICE,
        water_class_ids=settings.SEGFORMER_WATER_CLASS_IDS,
    )


@lru_cache(maxsize=1)
def get_detector():
    """Provide the IDetector implementation."""
    from backend.infrastructure.ai.detection.yolo_detector import YOLODetector
    return YOLODetector(
        model_name=settings.YOLO_MODEL_NAME,
        device=settings.DEVICE,
        confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold=settings.YOLO_IOU_THRESHOLD,
    )


@lru_cache(maxsize=1)
def get_storage_service():
    """Provide the IStorageService implementation."""
    from backend.infrastructure.storage.supabase_storage import SupabaseStorageService
    return SupabaseStorageService()


@lru_cache(maxsize=1)
def get_analysis_repository():
    """Provide the IAnalysisRepository implementation."""
    from backend.infrastructure.database.analysis_repository_impl import SupabaseAnalysisRepository
    return SupabaseAnalysisRepository()


@lru_cache(maxsize=1)
def get_report_generator():
    """Provide the IReportGenerator implementation."""
    from backend.infrastructure.reporting.pdf_generator import FPDFReportGenerator
    return FPDFReportGenerator()


def get_pipeline_orchestrator():
    """
    Provide the PipelineOrchestrator with all dependencies injected.
    Not cached because it holds references to cached singletons.
    """
    from backend.application.services.pipeline_orchestrator import PipelineOrchestrator
    return PipelineOrchestrator(
        preprocessor=get_preprocessor(),
        classifier=get_classifier(),
        segmenter=get_segmenter(),
        detector=get_detector(),
        storage=get_storage_service(),
        repository=get_analysis_repository(),
    )
