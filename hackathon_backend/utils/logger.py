"""
Logger Utility (utils/logger.py)
=================================
This file sets up a standardized logger for the application.
Having a central logger ensures all print statements follow the same format
and can be easily redirected to files or monitoring systems in production.
"""
import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger

# Create the main app logger
app_logger = setup_logger("hackathon_api")
