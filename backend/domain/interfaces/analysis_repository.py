"""
Analysis Repository Interface (Port)
=======================================
Abstract contract for persisting and retrieving Analysis entities.

This is the REPOSITORY PATTERN — the domain defines WHAT data operations
it needs, and the infrastructure provides HOW (Supabase, SQLite, etc.).
"""

from abc import ABC, abstractmethod

from backend.domain.entities.analysis import Analysis


class IAnalysisRepository(ABC):
    """
    Abstract repository for Analysis entities.

    Implementations:
      - SupabaseAnalysisRepository (production)
      - InMemoryAnalysisRepository (testing)
    """

    @abstractmethod
    async def save(self, analysis: Analysis) -> Analysis:
        """
        Persist a new analysis record.

        Args:
            analysis: The Analysis entity to save.

        Returns:
            The saved Analysis (with generated ID if not set).
        """
        ...

    @abstractmethod
    async def find_by_id(self, analysis_id: str) -> Analysis | None:
        """
        Find an analysis by its unique ID.

        Returns:
            The Analysis entity, or None if not found.
        """
        ...

    @abstractmethod
    async def update(self, analysis: Analysis) -> Analysis:
        """
        Update an existing analysis record.

        Args:
            analysis: The Analysis entity with updated fields.

        Returns:
            The updated Analysis entity.
        """
        ...

    @abstractmethod
    async def list_by_user(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[Analysis]:
        """
        List all analyses for a user, ordered by creation date (newest first).

        Args:
            user_id: The user's ID.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of Analysis entities.
        """
        ...

    @abstractmethod
    async def delete(self, analysis_id: str) -> bool:
        """
        Delete an analysis by ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def count_by_user(self, user_id: str) -> int:
        """Count total analyses for a user."""
        ...
