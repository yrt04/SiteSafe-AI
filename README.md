# SiteSafe AI

Real-time Personal Protective Equipment (PPE) compliance monitoring system for industrial and construction sites. The backend runs a fine-tuned 11-class YOLOv11n detector over a multithreaded OpenCV webcam pipeline, evaluates per-worker `NO_HELMET` and `NO_VEST` compliance via anatomical Region-of-Interest (ROI) spatial matching (`BCR >= 0.30` or expanded centroid inclusion), enforces a 10-second per-violation cooldown with automated JPEG evidence capture, and streams annotated frames and alert payloads over WebSockets (`ws://localhost:8765`) to a Next.js / Electron monitoring dashboard.

## Project Team

**Institution:** College of Computer Sciences and Information Technology (CCSIT), King Faisal University

- Yousef Alturki (222433741)
- Bishr Alghtam (222446763)
- Emad Albukhaytan (221441285)
- Mohammed Aldhamen (222421299)

**Supervisors:** Prof. Fawaz Alsaade (Supervisor), Dr. Ahmed Affifi (Co-Supervisor)

## Repository Layout

- `ws_server.py` - Async WebSocket video server, multithreaded camera reader, YOLOv11n inference loop, anatomical PPE compliance engine, 10s alert cooldown, and AVI/JPEG evidence capture.
- `detect_webcam.py` - Standalone OpenCV desktop viewer for local camera and model testing.
- `train.py` - Fine-tuning script (`yolo11n.pt` on `construction-ppe.yaml`, 50 epochs, `640x640`).
- `evaluate_milestone3.py` - Held-out test-split evaluation script generating per-class metrics, Core PPE mAP@50 analysis, and visual evidence under `report_evidence/`.
- `runs/detect/train/` - Fine-tuned production checkpoint (`weights/best.pt`), training configuration (`args.yaml`), and epoch metrics (`results.csv`).
- `report_evidence/` - Test-split confusion matrices, PR/F1 curves, sample annotated predictions, and structured metric summaries.
- `dashboard/` - Next.js 16 + React 19 + TypeScript real-time monitoring UI with optional Electron desktop wrapper (`main.js`).

## Prerequisites

- Python 3.10+ (CUDA-capable NVIDIA GPU recommended; falls back to CPU automatically)
- Node.js 20+ and npm

## Installation

### 1. Python Backend Environment

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

*(For CUDA acceleration on Windows/Linux, install the matching PyTorch CUDA wheel from https://pytorch.org/get-started/locally/ if needed.)*

### 2. Frontend Dashboard Dependencies

```powershell
cd dashboard
npm install
cd ..
```

## Running the System

The fine-tuned model checkpoint (`runs/detect/train/weights/best.pt`) is included in the repository and loaded automatically by default.

### Option A: WebSocket Backend + Next.js Web Dashboard

1. Start the Python WebSocket inference server from the repository root:

```powershell
.\venv\Scripts\python.exe ws_server.py
```

Optional CLI arguments:

```powershell
.\venv\Scripts\python.exe ws_server.py --port 8765 --camera 0 --model runs/detect/train/weights/best.pt
```

2. In a second terminal, start the Next.js development server:

```powershell
cd dashboard
npm run dev
```

Open `http://localhost:3000` in a browser and click **Start Camera Feed** to begin streaming and violation monitoring.

### Option B: Standalone Electron Desktop Application

`dashboard/main.js` builds the static Next.js export, spawns `ws_server.py` automatically using `..\venv\Scripts\python.exe`, and opens the desktop monitoring window:

```powershell
cd dashboard
npm run desktop
```

### Option C: Local OpenCV Debug Window (No Browser)

```powershell
.\venv\Scripts\python.exe detect_webcam.py
```

Press `S` to save an annotated snapshot to `screenshots/` or `Q` to quit.

## Training and Evaluation

To reproduce test-split metrics and regenerate plots in `report_evidence/` (requires the Ultralytics `construction-ppe` dataset under `datasets/construction-ppe/`):

```powershell
.\venv\Scripts\python.exe evaluate_milestone3.py
```

To retrain YOLOv11n from scratch:

```powershell
.\venv\Scripts\python.exe train.py
```

