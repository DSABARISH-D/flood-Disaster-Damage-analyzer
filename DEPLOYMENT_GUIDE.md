# Comprehensive Deployment Guide

This guide will walk you through deploying the complete Flood Damage Assessment stack (Frontend, Backend, AI, Database) to production environments using GitHub, Render, and Supabase.

---

## 1. Supabase (Database & Storage)

1.  Create a new project at [Supabase.com](https://supabase.com).
2.  Navigate to the **SQL Editor**.
3.  Copy the contents of `supabase/migrations/20260712_complete_schema.sql` and run it. This will automatically set up your Tables, Storage Buckets, and Row Level Security (RLS) policies.
4.  Navigate to **Project Settings -> API**.
5.  Copy the `Project URL` and `anon public key`. These will be your `SUPABASE_URL` and `SUPABASE_KEY` environment variables.

---

## 2. GitHub Preparation

1.  Initialize a git repository in the root folder: `git init`
2.  Create a `.gitignore` file. Ensure you ignore `temp_uploads/`, `__pycache__/`, `node_modules/`, and `.env`.
3.  Commit all files and push to a new GitHub repository.

---

## 3. Render Deployment (Backend & AI)

Render is ideal for the backend because it natively supports Docker and can handle the heavy PyTorch AI models.

**Option A: Using the Blueprint (Easiest)**
1.  Log in to [Render](https://render.com).
2.  Go to **Blueprints** -> **New Blueprint Instance**.
3.  Connect your GitHub repository.
4.  Render will read the `render.yaml` file in the root directory and automatically provision the Web Service.

**Option B: Manual Setup**
1.  Go to **New -> Web Service**.
2.  Connect your GitHub repo.
3.  Set the environment to **Docker**.
4.  Point the Dockerfile path to `Dockerfile.backend`.
5.  **CRITICAL**: Under Advanced, set your `SUPABASE_URL` and `SUPABASE_KEY` environment variables.
6.  **RAM Warning**: PyTorch models (especially SegFormer) consume significant RAM when loading. Choose a Render plan with at least 2GB of RAM (Starter/Standard plan).

---

## 4. Frontend Deployment (Vercel / Netlify)

While the `render.yaml` and `Dockerfile.frontend` can deploy the React app to Render, **Vercel** or **Netlify** are generally faster and free for static React apps.

1.  Log in to Vercel/Netlify.
2.  Import your GitHub repository.
3.  Set the **Root Directory** to `frontend/`.
4.  The Build Command should be `npm run build` and the Output Directory should be `dist`.
5.  Add an Environment Variable: `VITE_API_URL` pointing to your deployed Render Backend URL (e.g., `https://flood-backend.onrender.com`).
6.  Click Deploy!

---

## 🎉 Post-Deployment Verification

1.  Open your deployed Frontend URL.
2.  Upload a test satellite image.
3.  Verify the backend logs in Render to ensure the PyTorch models load correctly and the image uploads to Supabase Storage.
4.  Click the "Download PDF" button to ensure ReportLab generates the file correctly in the cloud environment.
