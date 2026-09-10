# 🚭 Smoking & Cigarette Detection in Restricted Areas

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

A real-time Computer Vision surveillance system that detects individuals smoking in restricted/non-smoking zones using a **dual-model YOLOv11 pipeline**, **ByteTrack multi-object tracking**, and an interactive **Streamlit web application**.

---

## 🚀 Live Demo & Streamlit Deployment

This repository is pre-configured for direct, zero-setup deployment on **Streamlit Community Cloud**.

### Option A: Deploy to Streamlit Cloud in 3 Steps

1. **Push to GitHub**:
   Upload this project directory to your GitHub account (see [GitHub Upload Guide](#-github-upload-guide) below).

2. **Log in to Streamlit**:
   Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.

3. **Deploy App**:
   - Click **New app**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Click **Deploy!** 🎉

---

## 📌 Project Overview

Traditional surveillance struggle to reliably detect smoking because cigarettes are tiny compared to human bodies. Standard Intersection over Union (IoU) metrics produce false negatives. 

This system resolves this challenge with:
- **YOLOv11s (COCO Pretrained)**: Detects persons in the frame (`class 0`).
- **Custom YOLOv11s (`best.pt`)**: Fine-tuned on annotated cigarette datasets to detect cigarettes (`class 1`).
- **Spatial Containment Algorithm**: Computes what percentage of a cigarette's bounding box is inside a person's bounding box.
- **ByteTrack Multi-Object Tracking**: Maintains persistent person IDs across video frames.
- **Streamlit Interactive UI**: Real-time image upload, video processing, live camera feed, adjustable confidence parameters, and Plotly analytics breakdown.

---

## 🎯 Detection Logic & Architecture

```
Video / Image / Camera Input
             │
             ├──► YOLOv11s (COCO) ────► Person Bounding Boxes [class 0]
             │                                │
             └──► YOLOv11s (Custom) ──► Cigarette Bounding Boxes [class 1]
                                              │
                              Spatial Containment Check
                (Is cigarette containment ratio >= 0.30 inside person box?)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
               Ratio ≥ 0.30                                    Ratio < 0.30
                      │                                               │
             🔴 SMOKING VIOLATION                             🟢 SAFE PERSON
             (Red Bounding Box)                              (Green Bounding Box)
```

$$\text{Containment Ratio} = \frac{\text{Area}(\text{Cigarette Box} \cap \text{Person Box})}{\text{Area}(\text{Cigarette Box})}$$

---

## 🗂️ Project Structure

```
ciggerete-detection/
├── app.py              # Streamlit Web Application (Main Deployment Entrypoint)
├── Inference.py        # Standalone OpenCV & ByteTrack CLI video pipeline
├── run_inference.py    # Simplified YOLO inference script
├── best.pt             # Custom fine-tuned cigarette detection weights
├── requirements.txt    # Python dependencies (Streamlit Cloud compatible)
├── packages.txt        # System OS APT dependencies for Streamlit Cloud
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation and Deployment Guide
```

---

## 💻 Local Installation & Usage

### 1. Clone or Download Repository
```bash
git clone https://github.com/YOUR_USERNAME/ciggerete-detection.git
cd ciggerete-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐙 GitHub Upload Guide

To make this project ready to deploy on Streamlit Cloud, run the following commands in your terminal inside this project directory:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Ready for Streamlit deployment"

# Rename branch to main
git branch -M main

# Link to your GitHub repository (replace URL with your own repo)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push code to GitHub
git push -u origin main
```

---

## ⚙️ Configuration Parameters

Adjustable parameters in the Streamlit Sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Person Confidence` | `0.40` | Minimum probability threshold for person detection |
| `Cigarette Confidence` | `0.25` | Minimum probability threshold for cigarette detection |
| `Containment Overlap` | `0.30` | Min fraction of cigarette box that must lie inside person box |
| `NMS IoU` | `0.45` | Non-Maximum Suppression IoU threshold |

---

## 🛠️ Tech Stack

- **Framework**: Python 3.10+, Streamlit 1.30+
- **Object Detection**: Ultralytics YOLOv11s
- **Computer Vision**: OpenCV
- **Visualization**: Plotly Express, Pandas, Pillow
- **Deployment**: Streamlit Community Cloud / Docker / HuggingFace Spaces

---

## 📄 License

This project is licensed under the MIT License.
