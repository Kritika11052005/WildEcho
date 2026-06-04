# 🍃 WildEcho: AI-Powered Bioacoustic Monitoring Platform

[![Next.js](https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-005C99?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)

WildEcho is a high-fidelity web platform designed to analyze passive acoustic monitoring (PAM) soundscapes and identify wildlife species in the Pantanal wetlands, South America. 

It is built as an ensemble pipeline using **Next.js (App Router + TypeScript)** in the frontend and a **FastAPI backend** running inference with ONNX Runtime models.

---

## 🏆 Kaggle Competition Case Study
This project incorporates the exact model ensemble designed for the **BirdCLEF+ 2026 Challenge** hosted by the **Cornell Lab of Ornithology**, **Google DeepMind**, and **iNaturalist**. 

* **Competitor:** Kritika Benjwal
* **Leaderboard Rank:** 1621 / 4,243 teams (Top 38%)
* **Private Score (ROC-AUC):** 0.93957

### Competition Notebooks:
* 📓 [BirdCLEF+ 2026 Inference Submission](https://www.kaggle.com/code/kritikabenjwal/birdclef26-inference-submission)
* 📓 [TaxaMoE Model Training Pipeline](https://www.kaggle.com/code/kritikabenjwal/birdclef26-taxamoe-training)

---

## 🧠 Model Ensemble Architecture

PAM datasets suffer from severe class bias (e.g., the training data is **97.9% bird vocalizations**). WildEcho addresses this imbalance using a hybrid architecture:

1. **Feature Extraction (Perch v2 ONNX):** Extracts robust 1536-dimensional embeddings from 12 consecutive 5-second audio windows.
2. **Taxa-aware Mixture of Experts (TaxaMoE):** Adapts the feature space and dynamically routes embeddings to specialized prototypical classification heads trained on specific taxonomic groups: *Aves, Insecta, Amphibia, Mammalia, and Reptilia*.
3. **Selective State Space Model (ProtoSSM):** Captures temporal transitions across audio windows to model sequence context.
4. **Sound Event Detection (SED):** Pinpoints the precise temporal boundaries and frame-level positions of vocal signals.
5. **Bayesian Prior Smoothing:** Adjusts logits using spatial (site) and temporal (hour) priors.
6. **Rank-based Axis Fusion:** Integrates predictions from all models using rank percentiles.

---

## 📁 Repository Structure

```text
├── backend/                  # FastAPI Web Server
│   ├── app.py                # Main API router, ONNX inference wrapper, and pipeline logic
│   ├── Dockerfile            # Hugging Face Spaces deployment container file
│   └── requirements.txt      # Python dependencies
├── frontend/                 # Next.js Frontend Application
│   ├── app/
│   │   ├── page.tsx          # Main Dashboard & Interactive Audio Workspace
│   │   ├── globals.css       # Core typography, dark theme tokens, and animations
│   │   └── layout.tsx        # SEO metadata and layout container
│   ├── public/               # Asset directory (backgrounds, generated HUD maps)
│   └── package.json          # Node dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Local Development Setup

### 1. Backend (FastAPI)
The backend loads model checkpoints from Hugging Face Hub using Git LFS or direct download.

```bash
# Navigate to backend folder
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install Python requirements
pip install -r requirements.txt

# Run server (default port 8000)
python -m uvicorn app:app --reload
```

### 2. Frontend (Next.js)
```bash
# Navigate to frontend folder
cd frontend

# Install Node modules
npm install

# Run local development server (port 3001)
npm run dev
```
Open [http://localhost:3001](http://localhost:3001) to interact with the dashboard.

---

## ☁️ Deployment

### Backend (Hugging Face Spaces)
The backend is set up to run in a Docker container on HF Spaces.
* Space Endpoint: `https://huggingface.co/spaces/Kritzzz11/bio-acoustica`

### Frontend (Vercel)
The Next.js frontend is optimized for deployment on Vercel:
1. Push the codebase to GitHub.
2. In Vercel, create a new project and import the repository.
3. Set the **Root Directory** to `frontend`.
4. Vercel automatically detects Next.js configurations and handles build generation.

---

## 👤 Credits & Contact
Created by **Kritika Benjwal**

* 🔗 [LinkedIn Profile](https://www.linkedin.com/in/kritika-benjwal)
* 🐙 [GitHub Profile](https://github.com/Kritika11052005)
* 🏆 [Kaggle Profile](https://www.kaggle.com/kritikabenjwal)
* 📧 [Email](mailto:ananya.benjwal@gmail.com)
