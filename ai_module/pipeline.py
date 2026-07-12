"""
Flood Assessment AI Pipeline
============================
This file orchestrates the flow of data through all three AI models.

Data Flow:
1. Input: Raw image bytes or PIL Image.
2. GATEKEEPER: ResNet50 checks if the image is flooded.
   -> If False, exit early (saves compute).
3. SEGMENTATION: SegFormer calculates the pixel-perfect flood mask and percentage.
4. DETECTION: YOLOv8 finds objects (Vehicles, Buildings).
5. INTERSECTION (Optional): Check which YOLO bounding boxes overlap with the SegFormer water mask to find *submerged* objects.
6. Output: Synthesized JSON dictionary.
"""
from PIL import Image
import numpy as np

from ai_module.models.resnet_classifier import ResNetClassifier
from ai_module.models.segformer_segmenter import SegformerSegmenter
from ai_module.models.yolo_detector import YOLODetector
from assessment_engine.engine import DamageAssessmentEngine

class FloodAssessmentPipeline:
    def __init__(self):
        print("Initializing AI Pipeline...")
        print("Loading ResNet50...")
        self.classifier = ResNetClassifier()
        print("Loading SegFormer...")
        self.segmenter = SegformerSegmenter()
        print("Loading YOLOv8...")
        self.detector = YOLODetector()
        print("Loading Assessment Engine...")
        self.assessor = DamageAssessmentEngine()
        print("AI Pipeline Ready!")

    def analyze(self, pil_image: Image.Image) -> dict:
        """
        Runs the full AI pipeline on the given image.
        """
        # Step 1: Classification (Gatekeeper)
        is_flood, confidence = self.classifier.predict(pil_image)
        
        if not is_flood:
            return {
                "flood": False,
                "confidence": confidence,
                "flood_percentage": 0,
                "severity": "Low",
                "objects": []
            }

        # Step 2: Segmentation
        water_mask, flood_percentage = self.segmenter.predict(pil_image)
        
        # Step 3: Object Detection
        detections = self.detector.predict(pil_image)
        
        # Step 4: Logic / Synthesis via Assessment Engine
        # The engine expects the raw mask and the detailed YOLO detections to calculate precise overlap
        report = self.assessor.generate_full_report(detections, water_mask)
        
        # Extract unique object names for the legacy response format
        unique_objects = [d["class"] for d in detections if "class" in d]
        unique_objects = list(set(unique_objects))
        
        return {
            "flood": is_flood,
            "confidence": confidence,
            "flood_percentage": report["metrics"]["flood_percentage"],
            "severity": report["assessment"]["severity_category"],
            "risk_level": report["assessment"]["risk_level"],
            "objects": unique_objects,
            "detailed_metrics": report["metrics"]
        }
