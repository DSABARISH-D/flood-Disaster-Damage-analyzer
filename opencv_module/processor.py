"""
OpenCV Image Processing Module
==============================
This module provides production-quality, highly-optimized image processing 
functions for the Flood Damage Assessment pipeline. It handles everything 
from basic preprocessing (resizing, normalizing) to complex visualization 
(overlays, bounding boxes).

Every function expects a valid numpy array (OpenCV image format, BGR or grayscale)
and returns a processed numpy array.
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Converts raw bytes to an OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640), keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resizes an image to the target dimensions.
    
    Why it's used: AI models (like YOLO or ResNet) require fixed input dimensions.
    
    Args:
        image: Input image (numpy array).
        target_size: Desired (width, height).
        keep_aspect_ratio: If True, pads the image to maintain original proportions.
        
    Returns:
        Resized image.
    """
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Calculate aspect ratio preserving dimensions
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of the target size and paste the resized image in the center
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normalizes image pixel values (typically for PyTorch models).
    
    Why it's used: Neural networks converge faster and perform better when 
    input data is normalized around a mean of 0 and standard deviation of 1.
    
    Args:
        image: Input BGR image (numpy array, uint8).
        mean: RGB mean values.
        std: RGB standard deviation values.
        
    Returns:
        Normalized image (float32 array).
    """
    # Convert BGR to RGB (OpenCV defaults to BGR, PyTorch expects RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Scale pixels to [0, 1] range
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Apply standard normalization: (pixel - mean) / std
    img_normalized = (img_float - np.array(mean)) / np.array(std)
    
    return img_normalized


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Why it's used: Satellite and drone imagery often suffers from poor lighting 
    or atmospheric haze. CLAHE improves local contrast, making features like 
    buildings and water boundaries pop out for the AI.
    
    Args:
        image: Input BGR image.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast-enhanced BGR image.
    """
    # Convert to LAB color space to isolate the lightness channel (L)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the Lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l_channel)
    
    # Merge the enhanced L channel back with A and B channels
    merged = cv2.merge((cl, a_channel, b_channel))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def remove_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Removes high-frequency noise from the image while preserving edges.
    
    Why it's used: Drone feeds can be noisy due to ISO settings or transmission artifacts. 
    A Gaussian blur or Bilateral filter smooths this out so the AI isn't confused by artifacts.
    
    Args:
        image: Input BGR image.
        kernel_size: Size of the blurring kernel (must be odd).
        
    Returns:
        Denoised BGR image.
    """
    # We use a Bilateral Filter because it's highly effective at removing noise
    # while keeping sharp edges (like building outlines), unlike a simple Gaussian blur.
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


def detect_edges(image: np.ndarray, lower_threshold: int = 50, upper_threshold: int = 150) -> np.ndarray:
    """
    Extracts structural edges from the image using the Canny Edge Detector.
    
    Why it's used: Edge detection can highlight roads, building footprints, 
    and sharp boundaries of floodwater, which can be fed as an auxiliary 
    input to certain segmentation models.
    
    Args:
        image: Input BGR image.
        lower_threshold: Canny lower bound.
        upper_threshold: Canny upper bound.
        
    Returns:
        Binary grayscale image containing edges (0 or 255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur before edge detection to prevent false positives from noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    return edges


def generate_flood_mask(image: np.ndarray, lower_blue: np.ndarray = np.array([90, 50, 50]), upper_blue: np.ndarray = np.array([130, 255, 255])) -> np.ndarray:
    """
    Generates a crude binary mask of water areas using HSV color thresholding.
    
    Why it's used: While SegFormer is our primary AI segmenter, a rapid HSV color 
    threshold can act as a highly efficient fallback or pre-filter for brown/blue water.
    
    Args:
        image: Input BGR image.
        lower_blue: Lower bound of water color in HSV.
        upper_blue: Upper bound of water color in HSV.
        
    Returns:
        Binary mask (grayscale numpy array).
    """
    # Convert BGR to HSV (Hue, Saturation, Value) for easier color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only water colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to remove tiny noise spots and fill holes in water
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 120, 255), alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a translucent colored mask (e.g., from SegFormer) onto the original image.
    
    Why it's used: Visualizing the AI's segmentation output for the end-user. 
    It paints the flooded areas with a semi-transparent blue hue.
    
    Args:
        image: Original BGR image.
        mask: Binary segmentation mask (1 for flood, 0 for background).
        color: BGR color to paint the mask (default is orange-ish/blue).
        alpha: Transparency level (0.0 to 1.0).
        
    Returns:
        Composited BGR image.
    """
    overlay = image.copy()
    
    # Ensure mask is boolean
    mask_bool = mask > 0
    
    # Apply the color only where the mask is True
    overlay[mask_bool] = color
    
    # Blend the overlay with the original image using the alpha weight
    # formula: result = original * (1 - alpha) + overlay * alpha
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Optional: Draw a solid contour line around the edge of the mask for better definition
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2) # Draw red outline
    
    return blended


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draws bounding boxes and labels for objects detected by YOLOv8.
    
    Why it's used: To show the user exactly where critical infrastructure 
    (buildings, vehicles) was detected within or near the flood zone.
    
    Args:
        image: Input BGR image.
        detections: List of dictionaries. Example: 
            [{"class": "Vehicles", "bbox": [x1, y1, x2, y2], "confidence": 0.95}]
            
    Returns:
        Image with bounding boxes drawn.
    """
    annotated = image.copy()
    
    # Color mapping for different object types (BGR format)
    colors = {
        "Buildings": (0, 255, 0),     # Green
        "Vehicles": (0, 165, 255),    # Orange
        "Roads": (255, 0, 0),         # Blue
        "default": (255, 255, 255)    # White
    }
    
    for obj in detections:
        label = obj.get("class", "Unknown")
        x1, y1, x2, y2 = map(int, obj.get("bbox", [0, 0, 0, 0]))
        conf = obj.get("confidence", 0.0)
        
        color = colors.get(label, colors["default"])
        
        # 1. Draw the bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        
        # 2. Prepare the label text
        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # 3. Calculate text background dimensions
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 4. Draw a solid rectangle behind the text for readability
        cv2.rectangle(annotated, (x1, y1 - text_h - baseline - 5), (x1 + text_w + 4, y1), color, thickness=cv2.FILLED)
        
        # 5. Draw the text (black text on colored background)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
        
    return annotated
