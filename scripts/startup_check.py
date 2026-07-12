#!/usr/bin/env python3
"""Startup check: Python version, venv, model availability, GPU.
Prints actionable fixes when problems detected.
"""
import sys
import os
import subprocess
from pathlib import Path

MIN_PY = (3, 8)
RECOMMENDED = (3, 11)

def check_python():
    v = sys.version_info
    print(f'Python version: {v.major}.{v.minor}.{v.micro}')
    if v.major == 3 and v.minor >= 14:
        print('WARNING: Python 3.14 is very new; recommend Python 3.11 or 3.12 for best compatibility')

def check_venv():
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        print('Virtualenv: active')
    else:
        print('Virtualenv: NOT active — consider creating and activating one')

def check_gpu():
    try:
        import torch
        print('Torch version:', torch.__version__)
        print('CUDA available:', torch.cuda.is_available())
    except Exception as e:
        print('Torch not available or failed to import:', e)

def check_models():
    from transformers import AutoConfig
    from models import segmenter
    from models import detector
    seg_id = os.environ.get('SEGFORMER_MODEL_ID') or 'nvidia/segformer-b0-finetuned-ade-512-512'
    print('Checking SegFormer model availability (will attempt to download if missing)...')
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        SegformerImageProcessor.from_pretrained(seg_id)
        SegformerForSemanticSegmentation.from_pretrained(seg_id)
        print('SegFormer: OK')
    except Exception as e:
        print('SegFormer: FAILED ->', e)

    print('Checking YOLOv8 via ultralytics...')
    try:
        from ultralytics import YOLO
        YOLO('yolov8n.pt')
        print('YOLOv8: OK (model available or downloaded)')
    except Exception as e:
        print('YOLOv8: FAILED ->', e)

def main():
    print('--- Startup check')
    check_python()
    check_venv()
    check_gpu()
    check_models()

if __name__ == '__main__':
    main()
