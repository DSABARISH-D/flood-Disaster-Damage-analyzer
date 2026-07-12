"""
FPDF Report Generator — Infrastructure Implementation
=======================================================
Implements IReportGenerator using fpdf2 for PDF creation.
"""

import tempfile
import os
from datetime import datetime

import numpy as np
from PIL import Image as PILImage
from fpdf import FPDF

from backend.domain.interfaces.report_generator_interface import IReportGenerator
from backend.domain.entities.damage_report import DamageReport


class FPDFReportGenerator(IReportGenerator):
    """
    Generates professional PDF damage assessment reports using fpdf2.
    """

    TITLE = "Disaster Damage Assessment Report"
    MARGIN = 15

    def generate_pdf(
        self,
        report: DamageReport,
        original_image: np.ndarray,
        segmentation_overlay: np.ndarray | None = None,
        detection_image: np.ndarray | None = None,
    ) -> bytes:
        """Generate a multi-page PDF report."""
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=self.MARGIN)
        pdf.set_title(self.TITLE)

        # Page 1: Summary
        pdf.add_page()
        self._header(pdf, "Assessment Summary")
        self._summary_section(pdf, report)

        # Page 2: Images
        pdf.add_page()
        self._header(pdf, "Visual Analysis")
        self._images_section(pdf, original_image, segmentation_overlay, detection_image)

        # Page 3: Detections + Recommendations
        pdf.add_page()
        self._header(pdf, "Findings & Recommendations")
        self._recommendations_section(pdf, report)

        return pdf.output()

    def _header(self, pdf: FPDF, subtitle: str = ""):
        """Styled header bar."""
        pdf.set_fill_color(30, 58, 95)
        pdf.rect(0, 0, 210, 28, "F")
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(self.MARGIN, 5)
        pdf.cell(0, 10, self.TITLE, align="L")

        if subtitle:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_xy(self.MARGIN, 16)
            pdf.cell(0, 8, subtitle, align="L")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(self.MARGIN, 16)
        pdf.cell(0, 8, datetime.now().strftime("%B %d, %Y  %I:%M %p"), align="R")

        pdf.set_text_color(0, 0, 0)
        pdf.set_y(33)

    def _summary_section(self, pdf: FPDF, report: DamageReport):
        """Key metrics and severity badge."""
        sev = report.severity
        cat = sev.category.value.title()

        # Severity badge
        colors = {"Minor": (255, 193, 7), "Moderate": (255, 152, 0),
                  "Severe": (244, 67, 54), "Critical": (106, 27, 154)}
        bg = colors.get(cat, (158, 158, 158))
        pdf.set_fill_color(*bg)
        txt_color = (255, 255, 255) if cat in ("Severe", "Critical") else (0, 0, 0)
        pdf.set_text_color(*txt_color)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 12, f"  {sev.emoji}  {cat}  |  Score: {sev.score:.1f}/10", fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Metrics table
        metrics = [
            ("Flood Area", f"{report.flood_area_percentage:.1f}%"),
            ("Objects Detected", str(report.total_objects)),
            ("In Flood Zone", str(report.objects_in_flood)),
            ("Severity Score", f"{sev.score:.1f}/10"),
            ("Category", cat),
        ]
        pdf.set_font("Helvetica", "", 10)
        for i, (label, value) in enumerate(metrics):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(235, 242, 252)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(90, 8, f"  {label}", border=1, fill=fill)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(90, 8, f"  {value}", border=1, fill=fill, ln=True)

    def _images_section(self, pdf: FPDF, original, overlay, detections):
        """Add analysis images."""
        images = [("Original", original), ("Flood Mask", overlay), ("Detections", detections)]
        temp_files = []

        for title, img_arr in images:
            if img_arr is None:
                continue
            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            PILImage.fromarray(img_arr.astype(np.uint8)).save(temp.name)
            temp_files.append(temp.name)

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, title, ln=True)
            pdf.image(temp.name, x=self.MARGIN, w=180)
            pdf.ln(5)

        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    def _recommendations_section(self, pdf: FPDF, report: DamageReport):
        """Add recommendations."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Recommendations", ln=True)
        pdf.ln(2)

        colors = {"CRITICAL": (220, 50, 50), "HIGH": (255, 152, 0),
                  "MEDIUM": (33, 150, 243), "LOW": (76, 175, 80), "INFO": (158, 158, 158)}

        for rec in report.recommendations:
            color = colors.get(rec.priority, (0, 0, 0))
            pdf.set_fill_color(*color)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(22, 7, f" {rec.priority}", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, f"  {rec.action}", ln=True)
            pdf.ln(1)
