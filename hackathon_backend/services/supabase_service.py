"""
Supabase API Integration Service
================================
This service acts as the bridge between our Python AI backend and the Supabase database.
It allows us to securely upload images, save predictions, log history, and fetch data.
"""
import os
from typing import List, Dict, Any
from supabase import create_client, Client

class SupabaseService:
    def __init__(self):
        """
        Initializes the Supabase client.
        Relies on environment variables SUPABASE_URL and SUPABASE_KEY.
        In a production environment, use the Service Role Key for backend admin bypass,
        or pass the user's JWT token for strict RLS enforcement.
        """
        url: str = os.environ.get("SUPABASE_URL", "")
        key: str = os.environ.get("SUPABASE_KEY", "")
        
        if not url or not key:
            print("WARNING: Supabase URL or Key is missing. Database operations will fail.")
            
        self.client: Client = create_client(url, key)

    def upload_image_to_storage(self, user_id: str, filename: str, file_bytes: bytes) -> str:
        """
        Uploads raw image bytes to the 'images' storage bucket.
        Path structure: {user_id}/{filename}
        """
        storage_path = f"{user_id}/{filename}"
        
        # Upload to Supabase Storage
        self.client.storage.from_("images").upload(
            file=file_bytes,
            path=storage_path,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        
        # Get the public URL or signed URL
        return storage_path

    def save_image_metadata(self, user_id: str, storage_path: str, filename: str, size: int) -> str:
        """
        Saves the image metadata to the `uploaded_images` table.
        Returns the generated image ID.
        """
        response = self.client.table("uploaded_images").insert({
            "user_id": user_id,
            "storage_path": storage_path,
            "original_filename": filename,
            "file_size_bytes": size
        }).execute()
        
        return response.data[0]["id"]

    def save_prediction_results(self, image_id: str, user_id: str, prediction_data: Dict[str, Any]) -> str:
        """
        Saves the high-level AI analysis to the `predictions` table.
        Returns the generated prediction ID.
        """
        response = self.client.table("predictions").insert({
            "image_id": image_id,
            "user_id": user_id,
            "is_flood": prediction_data.get("flood", False),
            "confidence_score": prediction_data.get("confidence", 0.0),
            "flood_percentage": prediction_data.get("flood_percentage", 0.0),
            "severity": prediction_data.get("severity", "Low"),
            "risk_level": prediction_data.get("risk_level", "")
        }).execute()
        
        return response.data[0]["id"]

    def save_detected_objects(self, prediction_id: str, objects: List[Dict[str, Any]]):
        """
        Batch inserts all YOLO detected objects into the `detected_objects` table.
        """
        if not objects:
            return
            
        rows_to_insert = []
        for obj in objects:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            rows_to_insert.append({
                "prediction_id": prediction_id,
                "object_class": obj.get("class", "Unknown"),
                "confidence": obj.get("confidence", 0.0),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "is_submerged": obj.get("is_submerged", False)
            })
            
        self.client.table("detected_objects").insert(rows_to_insert).execute()

    def log_user_history(self, user_id: str, action: str, details: Dict[str, Any] = None):
        """
        Appends an event to the audit `history` table.
        """
        self.client.table("history").insert({
            "user_id": user_id,
            "action_type": action,
            "details": details or {}
        }).execute()

# Instantiate a global service
supabase_db = SupabaseService()
