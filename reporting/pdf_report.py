"""
PDF Report Generator
=====================
Generates a professional PDF report with analysis results,
annotated images, damage metrics, and recommendations.
"""

import io
import os
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class PDFReportGenerator:
    """Generates a downloadable PDF damage assessment report using fpdf2."""

    def __init__(self):
        self.title = config.PDF_TITLE
        self.author = config.PDF_AUTHOR
        self.margin = config.PDF_MARGIN

    def generate(
        self,
        assessment_report: dict,
        original_image: np.ndarray,
        segmentation_overlay: np.ndarray = None,
        detection_image: np.ndarray = None,
        flood_mask: np.ndarray = None,
    ) -> bytes:
        """
        Generate a complete PDF report.

        Args:
            assessment_report: Output from DamageAssessmentEngine.assess()
            original_image: RGB numpy array of the original image
            segmentation_overlay: RGB image with flood mask overlay
            detection_image: RGB image with YOLO detections drawn

        Returns:
            bytes: PDF file content
        """
        from fpdf import FPDF

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=self.margin)
        pdf.set_author(self.author)
        pdf.set_title(self.title)

        # ── Page 1: Title & Summary ────────────────────────
        pdf.add_page()
        self._add_header(pdf)
        self._add_summary_section(pdf, assessment_report)

        # ── Page 2: Images ─────────────────────────────────
        pdf.add_page()
        self._add_header(pdf, "Visual Analysis")
        self._add_images_section(pdf, original_image, segmentation_overlay, detection_image)

        # ── Page 3: Detection Details & Score Breakdown ────
        pdf.add_page()
        self._add_header(pdf, "Detection Details")
        self._add_detection_table(pdf, assessment_report)
        self._add_score_breakdown(pdf, assessment_report)

        # ── Page 4: Recommendations ────────────────────────
        pdf.add_page()
        self._add_header(pdf, "Recommendations")
        self._add_recommendations(pdf, assessment_report)

        # ── Page 5: Charts ─────────────────────────────────
        pdf.add_page()
        self._add_header(pdf, "Statistical Analysis")
        self._add_charts(pdf, assessment_report)

        # Return as bytes
        return pdf.output()

    # ── Section Builders ───────────────────────────────────

    def _add_header(self, pdf, subtitle=None):
        """Add styled header to current page."""
        # Title bar
        pdf.set_fill_color(30, 58, 95)  # Dark navy
        pdf.rect(0, 0, 210, 30, "F")

        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(self.margin, 5)
        pdf.cell(0, 12, self.title, align="L")

        if subtitle:
            pdf.set_font("Helvetica", "", 11)
            pdf.set_xy(self.margin, 18)
            pdf.cell(0, 8, subtitle, align="L")

        # Date
        pdf.set_font("Helvetica", "", 9)
        pdf.set_xy(self.margin, 18 if not subtitle else 18)
        date_str = datetime.now().strftime("%B %d, %Y  %I:%M %p")
        pdf.cell(0, 8, date_str, align="R")

        pdf.set_text_color(0, 0, 0)
        pdf.set_y(35)

    def _add_summary_section(self, pdf, report: dict):
        """Add the key metrics summary."""
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Assessment Summary", ln=True)
        pdf.ln(2)

        # Severity card
        category = report.get("damage_category", "Unknown")
        severity = report.get("severity_score", 0)
        emoji = report.get("damage_category_emoji", "")

        color_map = {
            "Minor": (255, 193, 7),
            "Moderate": (255, 152, 0),
            "Severe": (244, 67, 54),
            "Critical": (33, 33, 33),
        }
        card_color = color_map.get(category, (158, 158, 158))

        pdf.set_fill_color(*card_color)
        text_color = (255, 255, 255) if category in ("Severe", "Critical") else (0, 0, 0)
        pdf.set_text_color(*text_color)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 15, f"  {emoji}  Damage Level: {category}  |  Score: {severity}/10", 
                 fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Key metrics table
        metrics = [
            ("Flood Detected", "Yes" if report.get("flood_detected") else "No"),
            ("Flood Confidence", f"{report.get('flood_confidence', 0):.1%}"),
            ("Flood Area Coverage", f"{report.get('flood_area_percentage', 0):.1f}%"),
            ("Total Objects Detected", str(report.get("total_objects_detected", 0))),
            ("Objects in Flood Zone", str(report.get("objects_in_flood_zone", 0))),
        ]

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(230, 240, 250)

        for i, (label, value) in enumerate(metrics):
            fill = i % 2 == 0
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(90, 8, f"  {label}", border=1, fill=fill)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(90, 8, f"  {value}", border=1, fill=fill, ln=True)

        pdf.ln(5)

        # Objects by class
        by_class = report.get("objects_by_class", {})
        if by_class:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Objects Detected by Class", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for cls, count in by_class.items():
                pdf.cell(0, 7, f"    {cls.title()}: {count}", ln=True)

    def _add_images_section(self, pdf, original, overlay, detections):
        """Add images to the PDF."""
        img_width = 85  # mm per image

        # Save images to temp files
        temp_files = []

        images = [
            ("Original Image", original),
            ("Flood Segmentation", overlay),
            ("Object Detection", detections),
        ]

        pdf.set_font("Helvetica", "B", 12)
        y_pos = pdf.get_y()

        for i, (title, img_array) in enumerate(images):
            if img_array is None:
                continue

            # Save to temp
            temp_path = self._save_temp_image(img_array)
            temp_files.append(temp_path)

            if i > 0 and i % 2 == 0:
                y_pos = pdf.get_y() + 5

            x_pos = self.margin + (i % 2) * (img_width + 5)

            pdf.set_xy(x_pos, y_pos)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(img_width, 6, title, ln=False)

            pdf.image(temp_path, x=x_pos, y=y_pos + 7, w=img_width)

            if i % 2 == 1:
                pdf.set_y(y_pos + img_width * 0.75 + 12)
                y_pos = pdf.get_y()

        # If odd number of images, advance y
        if len([img for _, img in images if img is not None]) % 2 == 1:
            pdf.set_y(y_pos + img_width * 0.75 + 12)

        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    def _add_detection_table(self, pdf, report: dict):
        """Add a table of all detected objects."""
        assessments = report.get("object_assessments", [])
        if not assessments:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "No objects detected.", ln=True)
            return

        pdf.set_font("Helvetica", "B", 10)
        headers = ["#", "Class", "Confidence", "Status", "Risk", "In Flood"]
        widths = [10, 30, 25, 35, 30, 25]

        # Header row
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        for header, w in zip(headers, widths):
            pdf.cell(w, 8, header, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Data rows
        pdf.set_font("Helvetica", "", 9)
        for i, obj in enumerate(assessments):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(245, 248, 255)

            risk_color = {
                "Critical": (220, 50, 50),
                "High": (255, 152, 0),
                "Low": (76, 175, 80),
            }.get(obj["risk_level"], (0, 0, 0))

            pdf.cell(widths[0], 7, str(i + 1), border=1, fill=fill)
            pdf.cell(widths[1], 7, obj["class"].title(), border=1, fill=fill)
            pdf.cell(widths[2], 7, f"{obj['confidence']:.0%}", border=1, fill=fill)
            pdf.cell(widths[3], 7, obj["status"], border=1, fill=fill)

            pdf.set_text_color(*risk_color)
            pdf.cell(widths[4], 7, obj["risk_level"], border=1, fill=fill)
            pdf.set_text_color(0, 0, 0)

            pdf.cell(widths[5], 7, "Yes" if obj["in_flood_zone"] else "No", 
                     border=1, fill=fill, ln=True)

        pdf.ln(5)

    def _add_score_breakdown(self, pdf, report: dict):
        """Add score breakdown section."""
        breakdown = report.get("score_breakdown", {})
        if not breakdown:
            return

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Severity Score Breakdown", ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(30, 58, 95)
        pdf.set_text_color(255, 255, 255)
        headers = ["Component", "Raw Value", "Score (/10)", "Weight", "Contribution"]
        widths = [40, 30, 25, 25, 30]
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 9)

        rows = [
            (
                "Flood Area",
                f"{breakdown['flood_area']['raw']}%",
                str(breakdown['flood_area']['score']),
                f"{breakdown['flood_area']['weight']:.0%}",
                str(breakdown['flood_area']['contribution']),
            ),
            (
                "Structures",
                str(breakdown['affected_structures']['count']),
                str(breakdown['affected_structures']['score']),
                f"{breakdown['affected_structures']['weight']:.0%}",
                str(breakdown['affected_structures']['contribution']),
            ),
            (
                "Object Types",
                "—",
                str(breakdown['object_types']['score']),
                f"{breakdown['object_types']['weight']:.0%}",
                str(breakdown['object_types']['contribution']),
            ),
        ]

        for i, row in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(245, 248, 255)
            for val, w in zip(row, widths):
                pdf.cell(w, 7, val, border=1, fill=fill)
            pdf.ln()

        # Total
        pdf.set_font("Helvetica", "B", 10)
        total = report.get("severity_score", 0)
        pdf.cell(sum(widths[:3]), 8, "TOTAL SEVERITY SCORE", border=1)
        pdf.cell(sum(widths[3:]), 8, f"{total} / 10", border=1, ln=True)

    def _add_recommendations(self, pdf, report: dict):
        """Add recommendations section."""
        recs = report.get("recommendations", [])

        priority_colors = {
            "CRITICAL": (220, 50, 50),
            "HIGH": (255, 152, 0),
            "MEDIUM": (33, 150, 243),
            "LOW": (76, 175, 80),
            "INFO": (158, 158, 158),
        }

        for rec in recs:
            priority = rec["priority"]
            color = priority_colors.get(priority, (0, 0, 0))

            pdf.set_fill_color(*color)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(25, 8, f"  {priority}", fill=True)

            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 8, f"  {rec['action']}", ln=True)
            pdf.ln(2)

    def _add_charts(self, pdf, report: dict):
        """Generate and add matplotlib charts to the PDF."""
        temp_files = []

        # Chart 1: Score breakdown pie
        chart_path = self._generate_score_pie(report)
        if chart_path:
            temp_files.append(chart_path)
            pdf.image(chart_path, x=self.margin, y=pdf.get_y(), w=80)

        # Chart 2: Object class bar chart
        chart_path2 = self._generate_class_bar(report)
        if chart_path2:
            temp_files.append(chart_path2)
            pdf.image(chart_path2, x=self.margin + 90, y=pdf.get_y(), w=80)

        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    # ── Chart Generators ───────────────────────────────────

    def _generate_score_pie(self, report: dict) -> str | None:
        """Generate severity score breakdown pie chart."""
        try:
            breakdown = report.get("score_breakdown", {})
            labels = ["Flood Area", "Structures", "Object Types"]
            values = [
                breakdown.get("flood_area", {}).get("contribution", 0),
                breakdown.get("affected_structures", {}).get("contribution", 0),
                breakdown.get("object_types", {}).get("contribution", 0),
            ]

            if sum(values) == 0:
                values = [1, 1, 1]  # Avoid empty pie

            colors = ["#3388ff", "#ff6b35", "#51cf66"]

            fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            wedges, texts, autotexts = ax.pie(
                values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"color": "white", "fontsize": 8}
            )
            for t in autotexts:
                t.set_fontsize(7)
                t.set_color("white")

            ax.set_title("Score Breakdown", color="white", fontsize=10, fontweight="bold")

            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(temp.name, bbox_inches="tight", facecolor="#1a1a2e")
            plt.close(fig)
            return temp.name
        except Exception:
            return None

    def _generate_class_bar(self, report: dict) -> str | None:
        """Generate detected objects bar chart."""
        try:
            by_class = report.get("objects_by_class", {})
            if not by_class:
                return None

            classes = list(by_class.keys())
            counts = list(by_class.values())

            fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")

            bars = ax.barh(classes, counts, color="#3388ff", edgecolor="#5599ff")
            ax.set_xlabel("Count", color="white", fontsize=9)
            ax.set_title("Detected Objects", color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(count), va="center", color="white", fontsize=8)

            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(temp.name, bbox_inches="tight", facecolor="#1a1a2e")
            plt.close(fig)
            return temp.name
        except Exception:
            return None

    # ── Helpers ────────────────────────────────────────────

    def _save_temp_image(self, img_array: np.ndarray) -> str:
        """Save numpy array as a temporary PNG file."""
        temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        pil_img.save(temp.name)
        return temp.name
