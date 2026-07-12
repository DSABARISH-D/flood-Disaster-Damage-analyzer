# Flood Damage Assessment using AI 🌊🚁

A complete, end-to-end AI platform for assessing flood damage from drone and satellite imagery. Built for speed, scale, and accuracy during emergency response scenarios.

## 🚀 Features

*   **Intelligent Gatekeeper (ResNet50)**: Instantly discards non-flood images to save compute.
*   **Semantic Segmentation (SegFormer)**: Pixel-perfect flood boundary mapping.
*   **Infrastructure Detection (YOLOv8)**: Real-time detection of submerged buildings, roads, and vehicles.
*   **Interactive Dashboard**: Live React interface with dynamic Recharts visualizations.
*   **Boardroom-Ready PDFs**: Automated ReportLab PDF generation with QR verification.
*   **Robust Backend**: FastAPI connected seamlessly to a Supabase Postgres Database & Storage.

## 🏗️ Architecture

```text
React (Frontend) ➔ FastAPI (Backend) ➔ PyTorch (AI) ➔ OpenCV (Vision) ➔ Supabase (Cloud) ➔ PDF Generator
```

## 🛠️ Local Development

### 1. Backend Setup
```bash
cd hackathon_backend
pip install -r requirements.txt
pip install -r ../ai_module/requirements.txt
```

Set up your `.env` file based on `.env.example`, then run the server:
```bash
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## ☁️ Deployment

This project is fully Dockerized and configured for Cloud deployment via Render.com and Supabase. Please see the `DEPLOYMENT_GUIDE.md` for step-by-step instructions.
