"""
Visualization Utilities
========================
Shared helpers for drawing overlays, bounding boxes,
comparison composites, and dashboard charts.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def draw_bboxes(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: RGB numpy array (H, W, 3)
        detections: List of detection dicts

    Returns:
        np.ndarray: Image with drawn bounding boxes
    """
    result = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]
        in_flood = det.get("in_flood_zone", False)

        # Color selection
        if in_flood:
            color = (255, 50, 50)  # Red for flood zone
        else:
            bgr_color = config.VIS_BBOX_COLORS.get(cls, config.VIS_BBOX_COLORS["default"])
            color = bgr_color  # Already in RGB for display

        thickness = config.VIS_BBOX_THICKNESS

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"{cls} {conf:.0%}"
        if in_flood:
            label += " FLOOD"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.VIS_FONT_SCALE, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            result, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, config.VIS_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA
        )

    return result


def create_flood_overlay(image: np.ndarray, flood_mask: np.ndarray, alpha: float = None) -> np.ndarray:
    """
    Create a blue flood overlay on the image.

    Args:
        image: RGB numpy array (H, W, 3)
        flood_mask: Binary mask (H, W) where 255 = flood
        alpha: Overlay transparency

    Returns:
        np.ndarray: Image with flood overlay
    """
    alpha = alpha or config.VIS_FLOOD_ALPHA
    overlay = image.copy()

    # Ensure mask matches image size
    if flood_mask.shape[:2] != image.shape[:2]:
        flood_mask = cv2.resize(flood_mask, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    # Create colored flood mask
    flood_color = np.array(config.VIS_FLOOD_COLOR, dtype=np.float64)
    mask_bool = flood_mask > 0

    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float64) * (1 - alpha) + flood_color * alpha
    ).astype(np.uint8)

    # Draw contours for crisp edges
    contours, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 80, 200), 2)

    return overlay


def create_side_by_side(image1: np.ndarray, image2: np.ndarray,
                         label1: str = "Original", label2: str = "Analyzed") -> np.ndarray:
    """Create a side-by-side comparison of two images."""
    # Ensure same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        image1 = cv2.resize(image1, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        image2 = cv2.resize(image2, (int(w2 * scale), target_h))

    # Create separator
    separator = np.ones((target_h, 4, 3), dtype=np.uint8) * 200

    # Concatenate
    composite = np.hstack([image1, separator, image2])

    # Add labels
    font_scale = 0.8
    cv2.putText(composite, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(composite, label2, (image1.shape[1] + 14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return composite


def generate_severity_gauge(score: float, category: str) -> bytes:
    """
    Generate a severity gauge chart as PNG bytes.

    Args:
        score: Severity score (0-10)
        category: Damage category string

    Returns:
        bytes: PNG image data
    """
    fig, ax = plt.subplots(figsize=(6, 2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Background bar
    ax.barh(0, 10, height=0.6, color="#1e2a3a", edgecolor="none")

    # Score bar with gradient effect
    color_map = {
        "Minor": "#ffc107",
        "Moderate": "#ff9800",
        "Severe": "#f44336",
        "Critical": "#9c27b0",
    }
    bar_color = color_map.get(category, "#757575")
    ax.barh(0, score, height=0.6, color=bar_color, edgecolor="none")

    # Score label
    ax.text(score + 0.2, 0, f"{score:.1f}", va="center", ha="left",
            fontsize=14, fontweight="bold", color="white")

    # Tick marks
    for i in range(11):
        ax.axvline(x=i, color="#2a3a4a", linewidth=0.5, ymin=0.2, ymax=0.8)

    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(11))
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#2a3a4a")

    ax.set_title(f"Severity: {category}", color="white", fontsize=12, fontweight="bold", pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_objects_chart(class_counts: dict) -> bytes:
    """Generate a horizontal bar chart of detected objects as PNG bytes."""
    if not class_counts:
        return None

    fig, ax = plt.subplots(figsize=(5, max(2, len(class_counts) * 0.6)), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#16213e")

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = ["#3388ff", "#51cf66", "#ff6b35", "#ffd43b", "#845ef7", "#20c997", "#ff8787"]
    bar_colors = [colors[i % len(colors)] for i in range(len(classes))]

    bars = ax.barh(classes, counts, color=bar_colors, edgecolor="none", height=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_xlabel("Count", color="white", fontsize=10)
    ax.set_title("Detected Objects", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines["bottom"].set_color("#2a3a4a")
    ax.spines["left"].set_color("#2a3a4a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.3 if counts else 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_score_breakdown_chart(breakdown: dict) -> bytes:
    """Generate the score breakdown donut chart."""
    try:
        labels = ["Flood Area", "Structures", "Object Types"]
        values = [
            breakdown.get("flood_area", {}).get("contribution", 0),
            breakdown.get("affected_structures", {}).get("contribution", 0),
            breakdown.get("object_types", {}).get("contribution", 0),
        ]

        if sum(values) == 0:
            values = [0.1, 0.1, 0.1]

        colors = ["#3388ff", "#ff6b35", "#51cf66"]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.8,
            textprops={"color": "white", "fontsize": 9}
        )

        # Donut center
        centre_circle = plt.Circle((0, 0), 0.55, fc="#0e1117")
        ax.add_artist(centre_circle)

        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight("bold")

        ax.set_title("Score Composition", color="white", fontsize=11, fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None
