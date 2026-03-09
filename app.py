"""
app.py - Main Flask Application
================================
Entry point for the Construction Material Detection System.
Handles routes, video streaming, and API endpoints.
"""

from flask import Flask, render_template, Response, jsonify, request
from detection.detect import ConstructionDetector
from database.db import DatabaseManager
import threading
import time
import logging
import os
import base64
import cv2
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename

# Allowed upload extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
UPLOAD_FOLDER      = 'static/uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']    = 'construction-ai-2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max upload

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Global State
# ─────────────────────────────────────────
detector = ConstructionDetector(model_path='model/construction_materials.pt')
db_manager = DatabaseManager()
detection_active = False
detection_lock = threading.Lock()

# ─────────────────────────────────────────
# Background DB Sync Thread
# ─────────────────────────────────────────
def sync_counts_to_db():
    """Periodically saves live counts to the database."""
    while True:
        time.sleep(5)
        if detection_active:
            counts = detector.get_counts()
            if counts:
                db_manager.update_counts(counts)

sync_thread = threading.Thread(target=sync_counts_to_db, daemon=True)
sync_thread.start()

# ─────────────────────────────────────────
# Video Frame Generator
# ─────────────────────────────────────────
def generate_frames(camera_id: int = 0):
    """
    Generator function that yields MJPEG frames.
    Runs YOLOv12 detection on each frame when active.
    """
    for frame_bytes in detector.stream_frames(camera_id, active_flag=lambda: detection_active):
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route('/')
def index():
    """Home page with live video stream and controls."""
    return render_template('index.html', classes=detector.CLASS_NAMES)


@app.route('/video_feed')
def video_feed():
    """MJPEG live stream endpoint."""
    camera_id = request.args.get('cam', 0, type=int)
    return Response(
        generate_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/dashboard')
def dashboard():
    """Material count dashboard with stock monitoring."""
    history = db_manager.get_history(limit=50)
    return render_template('dashboard.html', history=history, classes=detector.CLASS_NAMES)


@app.route('/get_counts')
def get_counts():
    """
    JSON API: Returns live material counts and system stats.
    Called by frontend JavaScript every second.
    """
    counts = detector.get_counts()
    stats  = detector.get_stats()
    alerts = _check_low_stock(counts)
    return jsonify({
        'counts': counts,
        'stats':  stats,
        'alerts': alerts,
        'active': detection_active,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Start or stop live detection."""
    global detection_active
    with detection_lock:
        detection_active = not detection_active
        logger.info(f"Detection {'started' if detection_active else 'stopped'}")
    return jsonify({'active': detection_active})


@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    """Reset all material counters to zero."""
    detector.reset_counts()
    db_manager.log_reset()
    return jsonify({'status': 'reset', 'message': 'All counts reset to 0'})


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Update low-stock threshold for a material."""
    data      = request.get_json()
    material  = data.get('material')
    threshold = data.get('threshold', 10)
    db_manager.set_threshold(material, threshold)
    return jsonify({'status': 'ok', 'material': material, 'threshold': threshold})


@app.route('/model_info')
def model_info():
    """JSON API: Returns which model / mode is currently active."""
    return jsonify({
        'model_info':  detector._model_info,
        'demo_mode':   detector.demo_mode,
        'use_mapping': getattr(detector, 'use_mapping', False),
        'conf':        detector.conf_threshold,
    })


def history():
    """JSON API: Returns count history for charting."""
    material = request.args.get('material', 'all')
    records  = db_manager.get_history(material=material, limit=100)
    return jsonify({'history': records})



# ─────────────────────────────────────────
# Image Upload Routes
# ─────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload")
def upload_page():
    """Image upload & detection page."""
    return render_template("upload.html", classes=detector.CLASS_NAMES)


@app.route("/detect_image", methods=["POST"])
def detect_image():
    """
    Receive an uploaded image, run YOLOv12 detection, return JSON:
      - base64-encoded annotated image
      - list of detections with class, confidence, bbox
      - per-class counts
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use JPG, PNG, BMP, or WEBP."}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        annotated, detections, counts = detector.detect_image(save_path)

        # Encode annotated image → base64 for browser
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64   = base64.b64encode(buffer).decode("utf-8")

        # Save result image for download
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
        cv2.imwrite(result_path, annotated)

        if counts:
            db_manager.update_counts(counts)

        return jsonify({
            "success":       True,
            "image_b64":     img_b64,
            "detections":    detections,
            "counts":        counts,
            "total":         len(detections),
            "result_file":   "result_" + filename,
        })

    except Exception as e:
        logger.error(f"Image detection error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _check_low_stock(counts: dict) -> list:
    """Compare current counts against configured thresholds."""
    thresholds = db_manager.get_thresholds()
    alerts = []
    for material, count in counts.items():
        limit = thresholds.get(material, 5)
        if count < limit:
            alerts.append({'material': material, 'count': count, 'threshold': limit})
    return alerts



# ─────────────────────────────────────────
# Batch Upload & Advanced Feature Routes
# ─────────────────────────────────────────

@app.route('/batch')
def batch_page():
    """Mega batch upload studio page."""
    return render_template('batch_upload.html', classes=detector.CLASS_NAMES)


@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    """
    Generate a detection density heatmap from a list of detections.
    Accepts JSON: { detections: [...], bg_b64: "...", alpha: 0.55 }
    Returns JSON: { heatmap_b64: "..." }
    """
    from utils.heatmap import build_heatmap
    import base64

    data       = request.get_json()
    detections = data.get('detections', [])
    bg_b64     = data.get('bg_b64', '')
    alpha      = float(data.get('alpha', 0.55))

    try:
        # Decode background image
        img_bytes  = base64.b64decode(bg_b64)
        nparr      = np.frombuffer(img_bytes, np.uint8)
        bg_img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bg_img is None:
            # fallback blank canvas
            bg_img = np.zeros((480, 640, 3), dtype=np.uint8)

        heatmap = build_heatmap(bg_img, detections, alpha=alpha)
        _, buf  = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64     = base64.b64encode(buf).decode('utf-8')
        return jsonify({'heatmap_b64': b64})
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """
    Generate a PDF report from batch detection results.
    Accepts JSON: { batch_results: [...], total_counts: {...} }
    Returns JSON: { pdf_url, filename }
    """
    from utils.pdf_report import generate_report
    import base64
    import uuid

    data          = request.get_json()
    batch_results = data.get('batch_results', [])
    total_counts  = data.get('total_counts', {})

    # Decode annotated images back to numpy for PDF embedding
    enriched = []
    for r in batch_results:
        b64 = r.get('image_b64', '')
        ann_img = None
        if b64:
            try:
                img_bytes = base64.b64decode(b64)
                nparr     = np.frombuffer(img_bytes, np.uint8)
                ann_img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                pass
        enriched.append({**r, 'annotated_img': ann_img})

    report_id   = uuid.uuid4().hex[:8]
    report_name = f"construction_report_{report_id}.pdf"
    report_path = os.path.join('static', 'reports', report_name)
    Path('static/reports').mkdir(parents=True, exist_ok=True)

    try:
        generate_report(
            batch_results  = enriched,
            total_counts   = total_counts,
            report_path    = report_path,
            project_name   = "Construction Site",
            session_id     = report_id,
        )
        return jsonify({
            'pdf_url':  '/' + report_path,
            'filename': report_name,
        })
    except Exception as e:
        logger.error(f"PDF report error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_analytics')
def get_analytics():
    """
    Return smart analytics from detection history + current counts.
    """
    from utils.analytics import compute_analytics
    history        = db_manager.get_history(limit=200)
    current_counts = detector.get_counts()
    analytics      = compute_analytics(history, current_counts)
    return jsonify(analytics)

# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == '__main__':
    logger.info("🏗️  Construction Material Detection System starting...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
