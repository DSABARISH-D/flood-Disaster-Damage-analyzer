"""
Configuration (config.py)
==========================
This file handles the configuration for our backend.
It uses Pydantic's BaseSettings to load environment variables and set defaults.
This makes our application configurable without changing code (e.g. changing ports, or model paths).
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Hackathon Flood Damage API"
    DEBUG: bool = True
    
    # AI Simulation (For hackathon demo purposes, we can simulate processing time)
    SIMULATE_AI_DELAY_SECONDS: float = 2.0
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    class Config:
        env_file = ".env"

# Create a global settings object to be imported across the app
settings = Settings()
