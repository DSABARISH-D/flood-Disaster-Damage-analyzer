"""
Image Processing Utility (utils/image_processing.py)
======================================================
This file isolates the image processing logic using OpenCV and Pillow (PIL).
When an image is uploaded, it comes as raw bytes. We need to convert those bytes
into an array of pixels (numpy array) that our AI models can understand.
"""
import io
import cv2
import numpy as np
from PIL import Image

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw uploaded bytes into an OpenCV BGR image (numpy array).
    """
    # Convert bytes to a numpy array of unsigned 8-bit integers
    np_arr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the array into an image using OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image bytes. Ensure it is a valid image file.")
        
    return img

def calculate_brightness(img: np.ndarray) -> float:
    """
    Example utility: Calculates the average brightness of the image.
    Could be used to reject images that are too dark (e.g., night time without IR).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
