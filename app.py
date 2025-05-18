from flask import Flask, request, render_template, jsonify, Response
import os
from datetime import datetime
import base64
import subprocess
import tempfile
import shutil
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route("/")
def index():
    labels = sorted([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))])
    models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")])
    return render_template("index.html", labels=labels, models=models)

@app.route("/add_label", methods=["POST"])
def add_label():
    label = request.json.get("label")
    if not label:
        return jsonify({"error": "Label is required"}), 400
    path = os.path.join(DATASET_DIR, label)
    os.makedirs(path, exist_ok=True)
    return jsonify({"success": True})

@app.route("/save_audio", methods=["POST"])
def save_audio():
    data = request.json
    label = data.get("label")
    if not label:
        return jsonify({"error": "Label is required"}), 400
    audio_data = base64.b64decode(data.get("audio").split(",")[1])
    folder = os.path.join(DATASET_DIR, label)
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_path = os.path.join(folder, f"{filename}.webm")
    wav_path = os.path.join(folder, f"{filename}.wav")
    with open(webm_path, "wb") as f:
        f.write(audio_data)
    subprocess.run(["ffmpeg", "-y", "-i", webm_path, wav_path], check=True)
    os.remove(webm_path)
    return jsonify({"success": True, "filename": f"{filename}.wav"})

@app.route("/train", methods=["POST"])
def train():
    def generate():
        process = subprocess.Popen(["bash", "run.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b""):
            yield line.decode()
    return Response(generate(), mimetype="text/plain")

@app.route("/infer", methods=["POST"])
def infer():
    if "model" not in request.form or "audio" not in request.files:
        return jsonify({"error": "Model and audio file are required"}), 400
    model_name = request.form.get("model")
    audio_file = request.files["audio"]
    tmpdir = tempfile.mkdtemp()
    try:
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(tmpdir, filename)
        audio_file.save(audio_path)
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            return jsonify({"error": "Selected model not found"}), 400
        result = subprocess.run(
            ["python3", "predict.py", "--model", model_path, "--audio", audio_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return jsonify({"error": "Inference failed", "details": result.stderr}), 500
        m = re.search(r"Predicted: (.+) \(confidence: ([\\d\\.]+)%\)", result.stdout)
        if not m:
            return jsonify({"error": "Unexpected output", "details": result.stdout}), 500
        label = m.group(1)
        confidence = float(m.group(2)) / 100.0
    finally:
        shutil.rmtree(tmpdir)
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

