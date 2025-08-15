#!/usr/bin/env python3
import os, sys, datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import jetson_inference
import jetson_utils

# ---------- Paths & folders ----------
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model & labels autodetect (can override with env vars) ----------
# Looks for resnet18.onnx (or any .onnx), and label.txt / labels.txt in project/
model_path = os.getenv("MODEL_PATH")
labels_path = os.getenv("LABELS_PATH")

if not model_path:
    candidates = [BASE_DIR / "resnet18.onnx"] + list(BASE_DIR.glob("*.onnx"))
    model_path = next((str(p) for p in candidates if p.exists()), None)

if not labels_path:
    candidates = [BASE_DIR / "label.txt", BASE_DIR / "labels.txt"]
    labels_path = next((str(p) for p in candidates if p.exists()), None)

if not model_path or not Path(model_path).exists():
    print(f"[ERROR] ONNX model not found.\n"
          f"Place your model in {BASE_DIR} (e.g., resnet18.onnx) or set MODEL_PATH=/abs/path/model.onnx")
    sys.exit(1)

if not labels_path or not Path(labels_path).exists():
    print(f"[ERROR] labels file not found.\n"
          f"Place label.txt or labels.txt in {BASE_DIR} or set LABELS_PATH=/abs/path/labels.txt")
    sys.exit(1)

INPUT_BLOB  = os.getenv("INPUT_BLOB", "input_0")
OUTPUT_BLOB = os.getenv("OUTPUT_BLOB", "output_0")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024)))  # 8MB default

print(f"[INFO] Using model:  {model_path}")
print(f"[INFO] Using labels: {labels_path}")
print(f"[INFO] Blobs: {INPUT_BLOB} / {OUTPUT_BLOB}")

# ---------- Load the network once ----------
try:
    net = jetson_inference.imageNet(argv=[
        sys.argv[0],
        f"--model={model_path}",
        f"--labels={labels_path}",
        f"--input_blob={INPUT_BLOB}",
        f"--output_blob={OUTPUT_BLOB}",
    ])
except Exception as e:
    print("[ERROR] imageNet failed to load network")
    print(e)
    sys.exit(1)

# ---------- Flask app ----------
app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/")
def index():
    # Serve static/index.html if it exists; otherwise simple message
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(app.static_folder, "index.html")
    return "Upload page not found. Create static/index.html", 200

@app.post("/api/classify")
def classify():
    if "image" not in request.files:
        return jsonify(ok=False, error="No file field named 'image'"), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(ok=False, error="Empty filename"), 400
    if not allowed_file(file.filename):
        return jsonify(ok=False, error="Unsupported file type"), 400

    safe_name = secure_filename(file.filename)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    in_path  = UPLOAD_DIR / f"{ts}_{safe_name}"
    out_path = OUTPUT_DIR / f"{in_path.stem}_labeled.jpg"

    # Save upload
    file.save(str(in_path))

    # Load & classify
    img = jetson_utils.loadImage(str(in_path))
    if img is None:
        return jsonify(ok=False, error="Failed to load image after upload"), 400

    class_idx, confidence = net.Classify(img)
    class_desc    = net.GetClassDesc(class_idx)
    class_network = net.GetNetworkName()

    # Overlay banner
    font = jetson_utils.cudaFont()
    banner = f"{safe_name} | {class_network} | {class_desc} ({confidence*100:.2f}%)"
    font.OverlayText(img, 10, 10, banner, color=(255,255,255,255), background=(0,0,0,160))

    # Save labeled image
    jetson_utils.saveImage(str(out_path), img)

    # URLs for browser
    input_url  = f"/static/uploads/{in_path.name}"
    output_url = f"/static/outputs/{out_path.name}"

    return jsonify(
        ok=True,
        label=class_desc,
        confidence=round(confidence * 100.0, 2),  # percentage
        class_idx=int(class_idx),
        network=class_network,
        input_image_url=input_url,
        output_image_url=output_url,
    )

@app.get("/api/ping")
def ping():
    return jsonify(status="ok")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
