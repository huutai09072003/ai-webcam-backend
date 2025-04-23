# 🧠 AI Webcam Backend (FastAPI + YOLOv8)

This is the backend service for the AI Webcam app. It uses **FastAPI** and **YOLOv8** (via `ultralytics`) to detect objects from base64-encoded images sent by the frontend.

---

## 🚀 Features

- ✅ Accepts image input as base64
- ✅ Runs object detection using a YOLOv8 model (`best.pt`)
- ✅ Returns detected label and annotated image (also in base64)
- ✅ CORS enabled — works smoothly with frontend

---

## 📦 Requirements

- Python 3.8+
- Virtualenv (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
