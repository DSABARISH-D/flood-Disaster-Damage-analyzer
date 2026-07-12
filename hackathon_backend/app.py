"""
FastAPI Entry Point (app.py)
=============================
This is the main application file that initializes FastAPI.
It sets up CORS (so our React frontend can talk to it) and mounts our API routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from hackathon_backend.config import settings
from hackathon_backend.api.routes import router as api_router
from hackathon_backend.utils.logger import app_logger

# Initialize the FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Backend for Flood Damage Assessment Hackathon Project",
    version="1.0.0"
)

# Set up CORS middleware to allow the frontend (running on a different port) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Allows specified origins (e.g., Vite's dev server)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all HTTP headers
)

# Mount the API router
# All routes in `api_router` will now be prefixed with `/api`
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Runs once when the server starts."""
    app_logger.info(f"Starting {settings.APP_NAME}...")

@app.get("/")
def read_root():
    """Health check endpoint at the root."""
    return {"message": "Welcome to the Flood Damage Assessment API", "status": "Online"}

# This block allows us to run the server by simply executing `python app.py`
if __name__ == "__main__":
    app_logger.info("Starting Uvicorn server...")
    uvicorn.run("hackathon_backend.app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
