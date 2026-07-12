"""
Report Generator Interface (Port)
====================================
Abstract contract for generating PDF/document reports.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from backend.domain.entities.damage_report import DamageReport


class IReportGenerator(ABC):
    """
    Abstract interface for report generation.

    Implementations:
      - FPDFReportGenerator (production — uses fpdf2)
      - HTMLReportGenerator (alternative — uses Jinja2)
    """

    @abstractmethod
    def generate_pdf(
        self,
        report: DamageReport,
        original_image: np.ndarray,
        segmentation_overlay: np.ndarray | None = None,
        detection_image: np.ndarray | None = None,
    ) -> bytes:
        """
        Generate a PDF report.

        Args:
            report: DamageReport entity with all assessment data.
            original_image: RGB numpy array of the uploaded image.
            segmentation_overlay: Optional overlay image with flood mask.
            detection_image: Optional image with detection bounding boxes.

        Returns:
            Raw PDF bytes ready for download.
        """
        ...
