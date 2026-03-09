# 🏗️ Construction Material Detection System
### Final Year Project — AI-Based Automatic Counting using YOLOv12 + Flask

---

## 📦 Project Structure

```
construction_detection/
├── app.py                        ← Flask entry point
├── requirements.txt
├── construction.yaml             ← YOLOv12 dataset config
│
├── model/
│   └── construction_materials.pt ← Trained YOLOv12 weights (place here)
│
├── detection/
│   ├── detect.py                 ← Main detector (YOLOv12 + ByteTrack)
│   ├── track.py                  ← Track lifecycle manager
│   └── counting.py               ← Counting engine with zone support
│
├── database/
│   └── db.py                     ← SQLite / MySQL persistence layer
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── templates/
│   ├── layout.html               ← Base navbar layout
│   ├── index.html                ← Live feed page
│   └── dashboard.html            ← Stock monitoring dashboard
│
└── utils/
    └── helpers.py                ← Shared utilities
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model
```
model/construction_materials.pt
```

### 3. Run the Flask app
```bash
python app.py
```
Open browser at **http://localhost:5000**

---

## 🎯 Material Classes
| ID | Class             | Description             |
|----|-------------------|-------------------------|
| 0  | cement_bag        | Bagged cement           |
| 1  | steel_bar         | Reinforcement steel      |
| 2  | steel_pipe        | Hollow steel pipes      |
| 3  | brick             | Standard clay bricks    |
| 4  | flyash_brick      | Flyash / eco bricks     |
| 5  | aac_block         | Aerated concrete blocks |
| 6  | formwork_shutter  | Wooden/metal shuttering |
| 7  | plastic_pipe      | PVC / plastic pipes     |

---

## 📡 API Endpoints

| Endpoint              | Method | Description                   |
|-----------------------|--------|-------------------------------|
| `/`                   | GET    | Live feed + controls          |
| `/video_feed`         | GET    | MJPEG camera stream           |
| `/dashboard`          | GET    | Stock monitoring dashboard    |
| `/get_counts`         | GET    | JSON: live counts + stats     |
| `/toggle_detection`   | POST   | Start / stop detection        |
| `/reset_counts`       | POST   | Reset all counts to 0         |
| `/set_threshold`      | POST   | Update low-stock threshold    |
| `/history`            | GET    | Historical count records      |

---

## 🏋️ Training Your Own Model

```bash
# Train from scratch
yolo detect train \
  data=construction.yaml \
  model=yolov12n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16

# Validate
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=construction.yaml

# Test detection on webcam
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=0 conf=0.5
```

Copy `best.pt` → `model/construction_materials.pt`.

---

## 🗄️ Database

**Default:** SQLite (no setup needed — auto-created at `database/construction.db`)

**MySQL (Production):** Edit `database/db.py`:
```python
db_manager = DatabaseManager(
    backend='mysql',
    mysql_config={
        'host':     'localhost',
        'user':     'root',
        'password': 'yourpassword',
        'database': 'construction_ai',
    }
)
```

---

## ⌨️ Keyboard Shortcuts

| Key       | Action             |
|-----------|--------------------|
| `Space`   | Start / Stop detection |
| `R`       | Reset counts       |

---

## 📊 System Architecture

```
Camera (Webcam / CCTV)
        │
        ▼
  OpenCV VideoCapture
        │
        ▼
  YOLOv12 Inference
  (bounding boxes + class IDs)
        │
        ▼
  ByteTrack (track_id per object)
        │
        ▼
  CountingEngine
  (unique ID → count once)
        │
        ├──► SQLite / MySQL  (persistence)
        │
        └──► Flask API (/get_counts)
                  │
                  ▼
           Browser Dashboard
           (live updates via JS polling)
```
