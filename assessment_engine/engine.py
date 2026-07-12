"""
Damage Assessment Engine
========================
This module calculates real-world metrics based on raw AI outputs.
It correlates the SegFormer flood mask with YOLO bounding boxes to determine
exactly what infrastructure is submerged, estimates human impact, and calculates
overall risk and severity levels.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class DamageAssessmentEngine:
    def __init__(self, avg_people_per_building: float = 3.5):
        """
        Initialize the engine with configuration parameters.
        
        Args:
            avg_people_per_building: Statistical average of occupants per structure 
                                     (used for population estimation).
        """
        self.avg_people_per_building = avg_people_per_building

    def calculate_flood_percentage(self, flood_mask: np.ndarray) -> float:
        """
        Calculates the exact percentage of the image covered by flood water.
        
        Algorithm:
        Total Water Pixels / Total Pixels * 100
        
        Args:
            flood_mask: Binary numpy array where 1 (or >0) represents water.
            
        Returns:
            Float percentage (0.0 to 100.0).
        """
        total_pixels = flood_mask.size
        if total_pixels == 0:
            return 0.0
            
        water_pixels = np.count_nonzero(flood_mask)
        percentage = (water_pixels / total_pixels) * 100.0
        
        return round(percentage, 2)

    def assess_infrastructure_damage(
        self, 
        detections: List[Dict[str, Any]], 
        flood_mask: np.ndarray,
        overlap_threshold: float = 0.2
    ) -> Dict[str, int]:
        """
        Determines exactly how many detected objects are actually flooded.
        """
        damage_counts = {}
        
        for obj in detections:
            raw_cls = obj.get("class")
            if not raw_cls:
                continue
                
            # Group classes dynamically based on COCO defaults
            if raw_cls in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                cls = "Vehicles"
            elif raw_cls == "person":
                cls = "People"
            elif raw_cls == "boat":
                cls = "Boats"
            else:
                cls = raw_cls.capitalize()
                
            if cls not in damage_counts:
                damage_counts[cls] = 0
                
            x1, y1, x2, y2 = map(int, obj.get("bbox", [0, 0, 0, 0]))
            
            # Ensure coordinates are within image bounds
            h, w = flood_mask.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            # Prevent zero-area boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract the ROI (Region of Interest) from the water mask
            roi = flood_mask[y1:y2, x1:x2]
            
            total_roi_pixels = roi.size
            if total_roi_pixels == 0:
                continue
            water_in_roi = np.count_nonzero(roi)
            
            # Check if the overlap exceeds the threshold
            if (water_in_roi / total_roi_pixels) > overlap_threshold:
                damage_counts[cls] += 1
                
        return damage_counts

    def estimate_population_affected(self, damaged_buildings_count: int) -> int:
        """
        Estimates the human population directly threatened by the flood.
        
        Algorithm:
        (Number of Damaged Buildings) * (Average People Per Building)
        
        Args:
            damaged_buildings_count: Number of flooded buildings.
            
        Returns:
            Estimated integer population.
        """
        population = int(damaged_buildings_count * self.avg_people_per_building)
        return population

    def calculate_damage_severity(
        self, 
        flood_percentage: float, 
        damage_counts: Dict[str, int], 
        population_affected: int
    ) -> Tuple[float, str]:
        """
        Calculates a numeric severity score (0 to 100) and assigns a categorical severity.
        
        Algorithm (Weighted Scoring):
        - Flood Coverage: 40% weight (e.g., 50% flood coverage = 20 points)
        - Population Affected: 30% weight (Capped at 50 people for max score)
        - Infrastructure: 30% weight (Buildings=5pts, Vehicles=2pts, Roads=1pt each)
        
        Args:
            flood_percentage: Pre-calculated flood area %.
            damage_counts: Dictionary of damaged objects.
            population_affected: Estimated human impact.
            
        Returns:
            Tuple of (Numeric Score 0-100, Categorical String).
        """
        # 1. Coverage Score (Max 40)
        score_coverage = (flood_percentage / 100.0) * 40.0
        
        # 2. Population Score (Max 30, assuming 50 people is max local disaster scale per image)
        score_population = min((population_affected / 50.0) * 30.0, 30.0)
        
        # 3. Infrastructure Score (Max 30)
        infra_raw = (
            (damage_counts.get("Buildings", 0) * 5) + 
            (damage_counts.get("Vehicles", 0) * 2) + 
            (damage_counts.get("Roads", 0) * 1)
        )
        score_infra = min(infra_raw, 30.0)
        
        # Total Score
        total_score = round(score_coverage + score_population + score_infra, 1)
        
        # Categorize
        if total_score >= 75:
            severity = "Severe"
        elif total_score >= 50:
            severity = "High"
        elif total_score >= 25:
            severity = "Medium"
        else:
            severity = "Low"
            
        return total_score, severity

    def determine_risk_level(self, severity: str, population_affected: int) -> str:
        """
        Translates the Damage Severity into actionable emergency response risk levels.
        
        Algorithm:
        Uses a decision matrix based on Severity and immediate human threat.
        
        Args:
            severity: Calculated severity category (Low, Medium, High, Severe).
            population_affected: Estimated human impact.
            
        Returns:
            String representing the Risk Level / Recommended Action.
        """
        if severity == "Severe":
            return "CRITICAL: Immediate Evacuation and Rescue Ops Required."
            
        if severity == "High":
            if population_affected > 0:
                return "HIGH RISK: Dispatch Rescue Teams to Submerged Structures."
            else:
                return "HIGH RISK: Severe Infrastructure Damage. Block Access Roads."
                
        if severity == "Medium":
            return "MODERATE RISK: Dispatch Assessment Teams. Monitor Water Levels."
            
        return "LOW RISK: Minor Flooding Detected. Continue Monitoring."

    def generate_full_report(self, detections: List[Dict], flood_mask: np.ndarray) -> Dict[str, Any]:
        """
        Utility function to run the entire engine pipeline and return a complete report.
        """
        flood_pct = self.calculate_flood_percentage(flood_mask)
        infra_damage = self.assess_infrastructure_damage(detections, flood_mask)
        
        buildings_damaged = infra_damage.get("Buildings", 0)
        roads_damaged = infra_damage.get("Roads", 0)
        vehicles_damaged = infra_damage.get("Vehicles", 0)
        
        population = self.estimate_population_affected(buildings_damaged)
        
        numeric_score, severity_category = self.calculate_damage_severity(flood_pct, infra_damage, population)
        risk_level = self.determine_risk_level(severity_category, population)
        
        return {
            "metrics": {
                "flood_percentage": flood_pct,
                "buildings_damaged": buildings_damaged,
                "roads_damaged": roads_damaged,
                "vehicles_damaged": vehicles_damaged,
                "estimated_population_affected": population
            },
            "assessment": {
                "severity_score_100": numeric_score,
                "severity_category": severity_category,
                "risk_level": risk_level
            }
        }
