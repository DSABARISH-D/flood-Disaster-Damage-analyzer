"""
Supabase Analysis Repository — Infrastructure Implementation
==============================================================
Implements IAnalysisRepository using Supabase PostgreSQL.
"""

from datetime import datetime
from typing import Any

from backend.domain.entities.analysis import Analysis
from backend.domain.enums.analysis_status import AnalysisStatus
from backend.domain.enums.damage_category import DamageCategory
from backend.domain.value_objects.severity import Severity
from backend.domain.value_objects.geo_coordinate import GeoCoordinate
from backend.domain.value_objects.image_metadata import ImageMetadata
from backend.domain.interfaces.analysis_repository import IAnalysisRepository
from backend.infrastructure.database.supabase_client import get_supabase_client


class SupabaseAnalysisRepository(IAnalysisRepository):
    """
    PostgreSQL-backed repository for Analysis entities via Supabase.
    """

    TABLE = "analyses"

    def __init__(self):
        self._client = get_supabase_client()

    async def save(self, analysis: Analysis) -> Analysis:
        row = self._to_row(analysis)
        self._client.table(self.TABLE).insert(row).execute()
        return analysis

    async def find_by_id(self, analysis_id: str) -> Analysis | None:
        result = (
            self._client.table(self.TABLE)
            .select("*")
            .eq("id", analysis_id)
            .maybe_single()
            .execute()
        )
        if result.data is None:
            return None
        return self._to_entity(result.data)

    async def update(self, analysis: Analysis) -> Analysis:
        row = self._to_row(analysis)
        self._client.table(self.TABLE).update(row).eq("id", analysis.id).execute()
        return analysis

    async def list_by_user(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[Analysis]:
        result = (
            self._client.table(self.TABLE)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return [self._to_entity(row) for row in (result.data or [])]

    async def delete(self, analysis_id: str) -> bool:
        result = self._client.table(self.TABLE).delete().eq("id", analysis_id).execute()
        return len(result.data or []) > 0

    async def count_by_user(self, user_id: str) -> int:
        result = (
            self._client.table(self.TABLE)
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        return result.count or 0

    # ── Mappers ────────────────────────────────────────

    @staticmethod
    def _to_row(analysis: Analysis) -> dict[str, Any]:
        """Convert Analysis entity → database row."""
        return {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "status": analysis.status.value,
            "image_url": analysis.image_url,
            "image_metadata": analysis.image_metadata.to_dict() if analysis.image_metadata else None,
            "latitude": analysis.location.lat if analysis.location else None,
            "longitude": analysis.location.lon if analysis.location else None,
            "classification_result": analysis.classification,
            "segmentation_result": {
                k: v for k, v in (analysis.segmentation or {}).items()
                if k != "flood_mask"
            } if analysis.segmentation else None,
            "detection_results": [d if isinstance(d, dict) else d for d in analysis.detections],
            "damage_assessment": analysis.score_breakdown,
            "severity_score": analysis.severity.score if analysis.severity else None,
            "damage_category": analysis.damage_category.value if analysis.damage_category else None,
            "annotated_image_url": analysis.annotated_image_url,
            "mask_image_url": analysis.mask_image_url,
            "error_message": analysis.error_message,
            "created_at": analysis.created_at.isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _to_entity(row: dict[str, Any]) -> Analysis:
        """Convert database row → Analysis entity."""
        analysis = Analysis(
            id=row["id"],
            user_id=row.get("user_id", ""),
            status=AnalysisStatus(row.get("status", "pending")),
            image_url=row.get("image_url", ""),
            classification=row.get("classification_result"),
            segmentation=row.get("segmentation_result"),
            detections=row.get("detection_results") or [],
            annotated_image_url=row.get("annotated_image_url", ""),
            mask_image_url=row.get("mask_image_url", ""),
            error_message=row.get("error_message", ""),
            score_breakdown=row.get("damage_assessment"),
        )

        # Reconstruct value objects
        if row.get("latitude") is not None and row.get("longitude") is not None:
            analysis.location = GeoCoordinate(lat=row["latitude"], lon=row["longitude"])

        if row.get("severity_score") is not None:
            analysis.severity = Severity(score=row["severity_score"])
            analysis.damage_category = DamageCategory.from_score(row["severity_score"])

        if row.get("image_metadata"):
            meta = row["image_metadata"]
            analysis.image_metadata = ImageMetadata(
                width=meta.get("width", 0), height=meta.get("height", 0),
                channels=meta.get("channels", 3), format=meta.get("format", "JPEG"),
                size_bytes=meta.get("size_bytes", 0), filename=meta.get("filename", ""),
            )

        if row.get("created_at"):
            analysis.created_at = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))

        return analysis
