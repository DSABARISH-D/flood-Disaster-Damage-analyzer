"""
Disaster Damage Analysis Pipeline — Global Configuration
=========================================================
Central configuration for model paths, thresholds, and pipeline parameters.
"""

import os

# ── Project Paths ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

# ── Model Configuration ───────────────────────────────────

# ResNet50 Flood Classifier
CLASSIFIER_MODEL_NAME = "resnet50"
CLASSIFIER_NUM_CLASSES = 2
CLASSIFIER_LABELS = ["no_flood", "flood"]
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5
CLASSIFIER_WEIGHTS_PATH = None  # Set to a .pth file path for custom fine-tuned weights

# SegFormer Flood Segmentation
SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
# Alternative flood-specific models (uncomment to use):
# SEGFORMER_MODEL_ID = "gdurkin/segformer-b0-finetuned-segments-floods-S2-pseudoRGBv1"
SEGFORMER_INPUT_SIZE = (512, 512)
# ADE20K class indices that represent water/flood
# 21 = water, 60 = river, 26 = sea
SEGFORMER_WATER_CLASS_IDS = [21, 60, 26]

# YOLOv8 Object Detection
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model for speed; use yolov8s.pt / yolov8m.pt for accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45
YOLO_INPUT_SIZE = 640
# COCO class IDs relevant to disaster assessment
YOLO_TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    8: "boat",
}
# Classes that represent structures (weighted higher in damage assessment)
YOLO_STRUCTURE_CLASSES = {"car", "bus", "truck", "boat"}
YOLO_PERSON_CLASSES = {"person"}

# ── Image Preprocessing ───────────────────────────────────
PREPROCESS_TARGET_SIZE = (640, 640)
PREPROCESS_NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
PREPROCESS_NORMALIZE_STD = [0.229, 0.224, 0.225]    # ImageNet std
PREPROCESS_DENOISE_STRENGTH = 10  # Bilateral filter strength

# ── Damage Assessment ─────────────────────────────────────
DAMAGE_SEVERITY_THRESHOLDS = {
    "minor": (0, 2.5),
    "moderate": (2.5, 5.0),
    "severe": (5.0, 7.5),
    "critical": (7.5, 10.0),
}
DAMAGE_WEIGHT_FLOOD_AREA = 0.40
DAMAGE_WEIGHT_AFFECTED_STRUCTURES = 0.30
DAMAGE_WEIGHT_OBJECT_TYPES = 0.30

# ── Visualization ─────────────────────────────────────────
VIS_FLOOD_COLOR = (0, 120, 255)       # Blue overlay for flood regions
VIS_FLOOD_ALPHA = 0.45                 # Overlay transparency
VIS_BBOX_COLORS = {
    "person": (255, 0, 0),             # Red
    "car": (0, 255, 0),                # Green
    "truck": (0, 200, 0),              # Dark green
    "bus": (0, 180, 50),               # Teal-green
    "boat": (255, 165, 0),             # Orange
    "bicycle": (255, 255, 0),          # Yellow
    "motorcycle": (200, 200, 0),       # Dark yellow
    "default": (255, 255, 255),        # White fallback
}
VIS_BBOX_THICKNESS = 2
VIS_FONT_SCALE = 0.6

# ── Map Defaults ──────────────────────────────────────────
MAP_DEFAULT_LOCATION = [20.5937, 78.9629]  # Center of India
MAP_DEFAULT_ZOOM = 5
MAP_FLOOD_COLOR = "#3388ff"
MAP_MARKER_COLOR = "red"

# ── PDF Report ────────────────────────────────────────────
PDF_TITLE = "Disaster Damage Assessment Report"
PDF_AUTHOR = "AI Damage Analysis System"
PDF_PAGE_WIDTH = 210  # A4 width in mm
PDF_PAGE_HEIGHT = 297  # A4 height in mm
PDF_MARGIN = 15
