"""
API Routes (api/routes.py)
===========================
End-to-end integration: Receives image -> Runs Real AI -> Generates PDF -> Saves to Supabase -> Returns JSON.
"""
import os
import sys
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

# Ensure the root project directory is in the Python path
# This prevents ModuleNotFoundError when running from inside hackathon_backend/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the strict Production AI Pipeline
from services.ai_pipeline import ProductionAIPipeline

from hackathon_backend.utils.pdf_generator import pdf_builder
from hackathon_backend.services.supabase_service import supabase_db
from hackathon_backend.utils.logger import app_logger

router = APIRouter()

# Initialize the heavy AI pipeline once when the module loads
try:
    ai_engine = ProductionAIPipeline()
except Exception as e:
    app_logger.error(f"Failed to load AI models: {e}")
    # We still instantiate the router, but requests will fail.
    
# Ensure a temporary directory exists for PDF image generation
os.makedirs("temp_uploads", exist_ok=True)

@router.post("/predict")
async def predict_flood_damage(
    ImageUpload: UploadFile = File(...),
    user_id: str = Form("anonymous-user-id")
):
    app_logger.info("==================================================")
    app_logger.info(f"New Request: {ImageUpload.filename} from {user_id}")
    
    if not ImageUpload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image file.")
        
    try:
        # 1. Read Image
        image_bytes = await ImageUpload.read()
        
        # 2. Save original image temporarily for PDF generation
        temp_id = str(uuid.uuid4())
        orig_path = f"temp_uploads/{temp_id}_orig.jpg"
        processed_path = f"temp_uploads/{temp_id}_processed.jpg"
        
        with open(orig_path, "wb") as f:
            f.write(image_bytes)
            
        # 3. Upload Original Image to Supabase Storage
        app_logger.info("Uploading original to Supabase Storage...")
        # (Assuming the method upload_image_to_storage exists)
        storage_url = supabase_db.upload_image_to_storage(user_id, f"{temp_id}_orig.jpg", image_bytes)
        
        # 4. Save Image Metadata to Database
        db_image_id = supabase_db.save_image_metadata(user_id, storage_url, ImageUpload.filename, len(image_bytes))
        
        # 5. Run REAL AI Pipeline (ResNet -> SegFormer -> YOLO -> Damage Engine)
        app_logger.info("Executing AI Pipeline...")
        prediction_result = ai_engine.run_pipeline(image_bytes)
        
        # Save processed image
        import cv2
        cv2.imwrite(processed_path, prediction_result.pop("cv2_image"))
        
        # Upload Processed image
        with open(processed_path, "rb") as f:
            processed_bytes = f.read()
        processed_url = supabase_db.upload_image_to_storage(user_id, f"{temp_id}_processed.jpg", processed_bytes)
        
        # 7. Generate PDF Report
        app_logger.info("Generating PDF Report...")
        
        # Backwards compatible formatting for the PDF generator
        legacy_result = {
            "flood": prediction_result["flood"],
            "severity": prediction_result["severity"],
            "confidence": prediction_result["confidence"] / 100.0,
            "flood_percentage": prediction_result["flood_percentage"],
            "objects": list(prediction_result["objects"].keys()),
            "id": db_image_id,
            "detailed_metrics": {
                "estimated_population_affected": prediction_result["people_at_risk"]
            }
        }
        pdf_bytes = pdf_builder.generate_report(orig_path, processed_path, legacy_result)
        pdf_filename = f"{temp_id}_report.pdf"
        pdf_url = f"https://flood-storage.supabase.co/reports/{user_id}/{pdf_filename}"
        
        # 9. Save AI Predictions to Database
        app_logger.info("Saving predictions to Supabase DB...")
        prediction_id = supabase_db.save_prediction_results(db_image_id, user_id, legacy_result)
        
        # 10. Log History
        supabase_db.log_user_history(user_id, "ANALYZE", {"prediction_id": prediction_id})
        
        app_logger.info("Request completed successfully!")
        
        # Final JSON mapping EXACTLY to requested schema
        return {
            "flood": prediction_result["flood"],
            "confidence": prediction_result["confidence"],
            "severity": prediction_result["severity"],
            "flood_percentage": prediction_result["flood_percentage"],
            "objects": prediction_result["objects"],
            "people_at_risk": prediction_result["people_at_risk"],
            "recommendation": prediction_result["recommendation"],
            "processed_image": processed_url,
            "pdf_report": pdf_url
        }
        
    except Exception as e:
        import traceback
        trace_details = traceback.format_exc()
        app_logger.error(f"Error during prediction: {e}\n{trace_details}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 12. Cleanup temp files reliably to prevent storage leaks
        try:
            if 'orig_path' in locals() and os.path.exists(orig_path): os.remove(orig_path)
            if 'processed_path' in locals() and os.path.exists(processed_path): os.remove(processed_path)
        except Exception as cleanup_e:
            app_logger.warning(f"Failed to cleanup temp files: {cleanup_e}")
