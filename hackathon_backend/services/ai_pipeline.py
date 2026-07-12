"""
AI Pipeline Service (services/ai_pipeline.py)
==============================================
This service orchestrates the AI models. It takes an image array, passes it 
through the models in the correct order, and synthesizes the final JSON response.
"""
import numpy as np
from hackathon_backend.services.models import ResNetClassifier, SegFormerSegmenter, YOLODetector
from hackathon_backend.utils.logger import app_logger

# Initialize models once when the app starts
classifier = ResNetClassifier()
segmenter = SegFormerSegmenter()
detector = YOLODetector()

def calculate_severity(flood_percentage: int, objects: list[str]) -> str:
    """Business logic to determine damage severity based on AI outputs."""
    if flood_percentage == 0:
        return "Low"
        
    score = 0
    if flood_percentage > 60:
        score += 3
    elif flood_percentage > 30:
        score += 2
    else:
        score += 1
        
    if "Buildings" in objects:
        score += 2
    if "Vehicles" in objects:
        score += 1
        
    if score >= 5:
        return "Severe"
    elif score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

def run_analysis_pipeline(image: np.ndarray) -> dict:
    """
    The main orchestration function.
    Runs the image through Classification -> Segmentation -> Detection.
    """
    app_logger.info("Starting AI pipeline analysis...")
    
    # 1. Classify
    is_flood, confidence = classifier.predict(image)
    app_logger.info(f"Classification: Flood={is_flood}, Confidence={confidence}%")
    
    # 2. Segment
    flood_percentage = segmenter.predict(image, is_flood)
    app_logger.info(f"Segmentation: Flood Coverage={flood_percentage}%")
    
    # 3. Detect
    objects = detector.predict(image, is_flood)
    app_logger.info(f"Detection: Objects Found={objects}")
    
    # 4. Synthesize Severity
    severity = calculate_severity(flood_percentage, objects)
    app_logger.info(f"Assessment: Severity={severity}")
    
    # Construct the exact JSON structure expected by the React frontend
    return {
        "flood": is_flood,
        "confidence": confidence,
        "flood_percentage": flood_percentage,
        "severity": severity,
        "objects": objects
    }
