# AI Model Weights Directory

This directory stores pre-trained and fine-tuned model weight files.

**These files are NOT committed to Git** (they are 100MB–500MB each).

## Download Weights

Run the setup script to download all required weights:

```bash
python scripts/download_weights.py
```

## Manual Download

| Model | File | Download From |
|-------|------|---------------|
| ResNet50 | `resnet50/best.pth` | Train on your flood dataset or use ImageNet weights (auto-downloaded) |
| SegFormer | Auto-downloaded | HuggingFace Hub (`nvidia/segformer-b0-finetuned-ade-512-512`) |
| YOLOv8 | `yolov8/yolov8n.pt` | [Ultralytics](https://github.com/ultralytics/assets/releases) (auto-downloaded) |

## Directory Structure

```
ai_weights/
├── resnet50/      # Fine-tuned ResNet50 weights (.pth)
├── segformer/     # SegFormer checkpoint (auto-cached by HuggingFace)
└── yolov8/        # YOLOv8 weights (.pt)
```
