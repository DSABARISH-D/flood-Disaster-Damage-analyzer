"""
Pipeline Orchestrator Service
===============================
The conductor of the entire analysis pipeline.

Coordinates the flow:
  Preprocess → Classify → (if flood) Segment → Detect → Assess → Save

This is the APPLICATION service that ties together:
  - Domain interfaces (IClassifier, ISegmenter, IDetector)
  - Use cases (AssessDamageUseCase)
  - Repository (IAnalysisRepository)
  - Storage (IStorageService)

It does NOT contain business logic itself — it delegates to use cases.
"""

import io
import traceback
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image as PILImage

from backend.domain.entities.analysis import Analysis
from backend.domain.value_objects.geo_coordinate import GeoCoordinate
from backend.domain.interfaces.preprocessor_interface import IPreprocessor
from backend.domain.interfaces.classifier_interface import IClassifier
from backend.domain.interfaces.segmenter_interface import ISegmenter
from backend.domain.interfaces.detector_interface import IDetector
from backend.domain.interfaces.analysis_repository import IAnalysisRepository
from backend.domain.interfaces.storage_interface import IStorageService
from backend.application.use_cases.assess_damage import AssessDamageUseCase
from backend.config.settings import settings


class PipelineOrchestrator:
    """
    Orchestrates the full AI analysis pipeline.

    Injected with all dependencies via constructor (Dependency Inversion).
    """

    def __init__(
        self,
        preprocessor: IPreprocessor,
        classifier: IClassifier,
        segmenter: ISegmenter,
        detector: IDetector,
        storage: IStorageService,
        repository: IAnalysisRepository,
    ):
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._segmenter = segmenter
        self._detector = detector
        self._storage = storage
        self._repository = repository
        self._assessor = AssessDamageUseCase()

    async def run(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        user_id: str = "",
        location: GeoCoordinate | None = None,
    ) -> Analysis:
        """
        Execute the full pipeline and return a completed Analysis entity.

        Steps:
          1. Create Analysis record (PENDING)
          2. Upload original image to storage
          3. Preprocess image
          4. Classify (flood / no-flood)
          5. If flood: segment → detect → assess
          6. Upload result images
          7. Save final results to DB
        """
        analysis = Analysis(user_id=user_id)

        try:
            # ── Step 1: Start ──────────────────────────
            analysis.start_processing()
            await self._repository.save(analysis)

            # ── Step 2: Upload original ────────────────
            image_path = f"{user_id}/{analysis.id}/original_{filename}"
            original_url = await self._storage.upload(
                bucket=settings.BUCKET_ORIGINAL_IMAGES,
                path=image_path,
                data=image_bytes,
                content_type=self._content_type(filename),
            )
            analysis.image_url = original_url

            # ── Step 3: Preprocess ─────────────────────
            preprocessed = self._preprocessor.preprocess(image_bytes, filename)
            analysis.image_metadata = preprocessed.metadata

            # Use GPS from EXIF or user-provided
            if location:
                analysis.location = location
            elif preprocessed.gps:
                analysis.location = preprocessed.gps

            # ── Step 4: Classify ───────────────────────
            classification = self._classifier.classify(preprocessed.pil_image)
            analysis.set_classification(classification.to_dict())

            # ── Step 5: Flood Pipeline ─────────────────
            if classification.is_flood:
                # Segment
                segmentation = self._segmenter.segment(preprocessed.pil_image)
                analysis.set_segmentation(segmentation.to_dict())

                # Upload mask overlay
                overlay = self._create_flood_overlay(preprocessed.display_rgb, segmentation.flood_mask)
                mask_bytes = self._numpy_to_bytes(overlay)
                mask_url = await self._storage.upload(
                    bucket=settings.BUCKET_RESULT_IMAGES,
                    path=f"{user_id}/{analysis.id}/flood_mask.png",
                    data=mask_bytes,
                )
                analysis.mask_image_url = mask_url

                # Detect
                detection = self._detector.detect(preprocessed.display_rgb, segmentation.flood_mask)
                analysis.set_detections([d.to_dict() for d in detection.detections])

                # Upload annotated image
                if detection.annotated_image is not None:
                    ann_bytes = self._numpy_to_bytes(detection.annotated_image)
                    ann_url = await self._storage.upload(
                        bucket=settings.BUCKET_RESULT_IMAGES,
                        path=f"{user_id}/{analysis.id}/annotated.png",
                        data=ann_bytes,
                    )
                    analysis.annotated_image_url = ann_url

                # Assess
                report = self._assessor.execute(classification, segmentation, detection)
                analysis.complete(
                    severity=report.severity,
                    recommendations=[r.to_dict() for r in report.recommendations],
                )
                analysis.score_breakdown = report.to_dict()
            else:
                # No flood — complete without segmentation
                from backend.domain.value_objects.severity import Severity
                analysis.complete(severity=Severity(score=0.0), recommendations=[])

            # ── Step 6: Save final ─────────────────────
            await self._repository.update(analysis)
            return analysis

        except Exception as e:
            analysis.fail(error_message=f"{type(e).__name__}: {str(e)}")
            try:
                await self._repository.update(analysis)
            except Exception:
                pass
            raise

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    def _create_flood_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Create blue flood overlay on image."""
        overlay = image.copy()
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        flood_color = np.array([0, 120, 255], dtype=np.float64)
        mask_bool = mask > 0
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float64) * (1 - alpha) + flood_color * alpha).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 80, 200), 2)
        return overlay

    @staticmethod
    def _numpy_to_bytes(img: np.ndarray) -> bytes:
        """Convert numpy RGB array to PNG bytes."""
        pil = PILImage.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _content_type(filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpeg"
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "tiff": "image/tiff"}.get(ext, "image/jpeg")
