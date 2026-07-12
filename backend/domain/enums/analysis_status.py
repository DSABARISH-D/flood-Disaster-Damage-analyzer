"""
Analysis Status Enum
=====================
Tracks the lifecycle of an analysis through the pipeline.

States:
  PENDING    → created, waiting to start
  PROCESSING → currently running through AI models
  COMPLETED  → all stages finished successfully
  FAILED     → an error occurred during processing
"""

from enum import Enum


class AnalysisStatus(str, Enum):
    """
    Lifecycle status of a damage analysis.

    Inherits from `str` so it serializes cleanly to JSON.

    Usage:
        status = AnalysisStatus.PROCESSING
        print(status.value)  # "processing"
        print(status == "processing")  # True (because str mixin)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether this status represents a final state (no more transitions)."""
        return self in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)

    @property
    def is_active(self) -> bool:
        """Whether this status represents an active/in-progress state."""
        return self == AnalysisStatus.PROCESSING
