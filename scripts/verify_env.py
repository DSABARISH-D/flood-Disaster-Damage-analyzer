#!/usr/bin/env python3
"""Verify key Python packages and print versions.
Usage: python scripts/verify_env.py
"""
import importlib
import sys

packages = [
    'torch', 'torchvision', 'transformers', 'huggingface_hub', 'safetensors',
    'ultralytics', 'cv2', 'PIL', 'numpy', 'streamlit', 'tensorflow',
    'matplotlib', 'pandas', 'skimage', 'reportlab', 'fpdf', 'requests'
]

ok = True
for pkg in packages:
    try:
        m = importlib.import_module(pkg)
        version = getattr(m, '__version__', None)
        print(f"OK: {pkg} version={version}")
    except Exception as exc:
        print(f"MISSING/ERROR: {pkg} -> {exc}")
        ok = False

if not ok:
    print('\nOne or more packages are missing or failed to import.')
    print('Run: python -m pip install -r requirements.txt')
    sys.exit(2)
else:
    print('\nAll checks passed.')
