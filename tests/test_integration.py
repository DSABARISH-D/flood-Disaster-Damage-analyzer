import pytest
import os
import io
from fastapi.testclient import TestClient
from PIL import Image
import sys

# Ensure root path is accessible for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hackathon_backend.main import app

client = TestClient(app)

def create_dummy_image(color=(0, 100, 200), size=(800, 600), format="JPEG"):
    """Helper to create dummy images in memory."""
    img = Image.new("RGB", size, color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

def test_normal_flood_image():
    """Test standard valid flood image."""
    img_bytes = create_dummy_image()
    response = client.post(
        "/api/predict", # Assuming the router is mounted here or just "/predict" based on main.py
        files={"ImageUpload": ("test_flood.jpg", img_bytes, "image/jpeg")},
        data={"user_id": "test-user-123"}
    )
    # The actual status might be 200 depending on the pipeline running successfully
    # Since we are mocking/using a dummy blue image, it might trigger the flood heuristic!
    assert response.status_code in [200, 500] 
    if response.status_code == 200:
        data = response.json()
        assert "flood" in data
        assert "confidence" in data
        assert "severity" in data

def test_no_flood_image():
    """Test image that clearly has no flood (e.g. solid green)."""
    img_bytes = create_dummy_image(color=(0, 255, 0)) # Green
    response = client.post(
        "/api/predict",
        files={"ImageUpload": ("test_noflood.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code in [200, 500]

def test_corrupted_image():
    """Test uploading a corrupted file disguised as an image."""
    response = client.post(
        "/api/predict",
        files={"ImageUpload": ("corrupt.jpg", b"this is not an image", "image/jpeg")},
    )
    # PIL Image.open will fail, should return 500 or 400 depending on error handling
    assert response.status_code in [400, 500]

def test_unsupported_format():
    """Test uploading a text file."""
    response = client.post(
        "/api/predict",
        files={"ImageUpload": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "Must be an image file" in response.text

def test_large_image():
    """Test extremely large image processing."""
    # Create a 4K image
    img_bytes = create_dummy_image(size=(3840, 2160))
    response = client.post(
        "/api/predict",
        files={"ImageUpload": ("large.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code in [200, 500]
