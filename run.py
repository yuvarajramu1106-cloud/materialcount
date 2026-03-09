#!/usr/bin/env python3
"""
run.py — Quick launcher for the Construction AI Detection System
================================================================
Run this instead of app.py for a friendlier startup experience.
Checks dependencies, prints status, then launches Flask.
"""

import sys
import os

print("""
╔══════════════════════════════════════════════════════════╗
║   🏗️  Construction Material Detection System             ║
║   YOLOv12 + ByteTrack + Flask                           ║
║   Final Year Project — AI/ML                            ║
╚══════════════════════════════════════════════════════════╝
""")

# ── Check Python version ──────────────────────────────────
if sys.version_info < (3, 9):
    print("❌  Python 3.9+ required. Current:", sys.version)
    sys.exit(1)
print(f"✅  Python {sys.version.split()[0]}")

# ── Check dependencies ────────────────────────────────────
missing = []
checks = {
    'flask':        'Flask',
    'cv2':          'OpenCV (opencv-python)',
    'ultralytics':  'Ultralytics (YOLOv12)',
    'numpy':        'NumPy',
    'reportlab':    'ReportLab (PDF reports)',
}
for module, name in checks.items():
    try:
        __import__(module)
        print(f"✅  {name}")
    except ImportError:
        print(f"❌  {name} — NOT INSTALLED")
        missing.append(module)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt\n")
    ans = input("Continue anyway? (y/n): ").strip().lower()
    if ans != 'y':
        sys.exit(1)

# ── Check model ───────────────────────────────────────────
from pathlib import Path
model_path = 'model/construction_materials.pt'
if Path(model_path).exists():
    print(f"\n🎯  Custom model found: {model_path}")
else:
    print(f"\n⚠️  Custom model not found at: {model_path}")
    print("   → Base yolov12n.pt will be auto-downloaded (~6 MB)")
    print("   → COCO→Construction class mapping will be active")

# ── Check camera ──────────────────────────────────────────
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅  Camera (device 0) accessible")
        cap.release()
    else:
        print("⚠️  Camera not detected — live feed will show error frame")
except Exception:
    pass

# ── Launch ────────────────────────────────────────────────
print("""
──────────────────────────────────────────────────────────
  Starting Flask server...
  Open your browser at:  http://localhost:5000
  Press Ctrl+C to stop
──────────────────────────────────────────────────────────
""")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from app import app
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
