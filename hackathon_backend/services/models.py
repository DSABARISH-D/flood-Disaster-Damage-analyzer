"""
AI Models Interfaces (services/models.py)
==========================================
This file represents the individual AI models (ResNet50, SegFormer, YOLOv8).
For a hackathon where fast deployment is key, these classes can either wrap the 
actual PyTorch inference code or act as sophisticated mocks based on image properties.
Here, we implement them as mock services that simulate real AI analysis.
"""
import random
import numpy as np
from typing import List, Tuple

class ResNetClassifier:
    """Classifies if the image contains flooding (Yes/No) and the confidence."""
    
    def predict(self, image: np.ndarray) -> Tuple[bool, int]:
        # In a real scenario, this would run: torch.argmax(model(image))
        # For the hackathon, we simulate a realistic response.
        
        is_flood = random.choice([True, True, True, False]) # 75% chance of flood for demo
        confidence = random.randint(85, 99) if is_flood else random.randint(90, 99)
        
        return is_flood, confidence


class SegFormerSegmenter:
    """Segments the flood water and calculates the percentage of the image covered."""
    
    def predict(self, image: np.ndarray, is_flood: bool) -> int:
        # In a real scenario, this would count the water pixels / total pixels.
        if not is_flood:
            return 0
            
        flood_percentage = random.randint(30, 85)
        return flood_percentage


class YOLODetector:
    """Detects objects in the image (Buildings, Roads, Vehicles)."""
    
    def predict(self, image: np.ndarray, is_flood: bool) -> List[str]:
        # In a real scenario, this runs YOLO inference and extracts bounding box labels.
        if not is_flood:
            return []
            
        possible_objects = ["Buildings", "Roads", "Vehicles", "Infrastructure"]
        # Randomly detect 1 to 3 object types
        num_objects = random.randint(1, 3)
        detected = random.sample(possible_objects, num_objects)
        
        return detected
