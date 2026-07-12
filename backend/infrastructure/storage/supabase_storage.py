"""
Supabase Storage Service — Infrastructure Implementation
=========================================================
Implements IStorageService using Supabase Storage (S3-compatible).
"""

from backend.domain.interfaces.storage_interface import IStorageService
from backend.infrastructure.database.supabase_client import get_supabase_client


class SupabaseStorageService(IStorageService):
    """
    File storage service backed by Supabase Storage buckets.
    """

    def __init__(self):
        self._client = get_supabase_client()

    async def upload(
        self, bucket: str, path: str, data: bytes, content_type: str = "image/png"
    ) -> str:
        """Upload file and return its public URL."""
        self._client.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type},
        )
        return await self.get_public_url(bucket, path)

    async def download(self, bucket: str, path: str) -> bytes:
        """Download file bytes from storage."""
        response = self._client.storage.from_(bucket).download(path)
        return response

    async def delete(self, bucket: str, path: str) -> bool:
        """Delete a file from storage."""
        try:
            self._client.storage.from_(bucket).remove([path])
            return True
        except Exception:
            return False

    async def get_public_url(self, bucket: str, path: str) -> str:
        """Get the public URL for a stored file."""
        result = self._client.storage.from_(bucket).get_public_url(path)
        return result
