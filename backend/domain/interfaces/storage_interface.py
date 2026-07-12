"""
Storage Service Interface (Port)
==================================
Abstract contract for file storage operations (images, PDFs).
"""

from abc import ABC, abstractmethod


class IStorageService(ABC):
    """
    Abstract interface for file storage.

    Implementations:
      - SupabaseStorage (production)
      - LocalFileStorage (development/testing)
    """

    @abstractmethod
    async def upload(
        self, bucket: str, path: str, data: bytes, content_type: str = "image/png"
    ) -> str:
        """
        Upload a file to storage.

        Args:
            bucket: Storage bucket name (e.g., "original-images").
            path: File path within the bucket.
            data: Raw file bytes.
            content_type: MIME type of the file.

        Returns:
            Public URL of the uploaded file.
        """
        ...

    @abstractmethod
    async def download(self, bucket: str, path: str) -> bytes:
        """
        Download a file from storage.

        Returns:
            Raw file bytes.
        """
        ...

    @abstractmethod
    async def delete(self, bucket: str, path: str) -> bool:
        """
        Delete a file from storage.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def get_public_url(self, bucket: str, path: str) -> str:
        """
        Get the public URL for a file in storage.

        Returns:
            Publicly accessible URL string.
        """
        ...
