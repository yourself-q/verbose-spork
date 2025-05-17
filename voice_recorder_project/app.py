from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime
import base64
import wave
import subprocess

app = Flask(__name__)

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

@app.route("/")
def index():
    labels = sorted([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))])
    return render_template("index.html", labels=labels)

@app.route("/add_label", methods=["POST"])
def add_label():
    label = request.json["label"]
    path = os.path.join(DATASET_DIR, label)
    os.makedirs(path, exist_ok=True)
    return jsonify({"success": True})

@app.route("/save_audio", methods=["POST"])
def save_audio():
    data = request.json
    label = data["label"]
    audio_data = base64.b64decode(data["audio"].split(",")[1])
    folder = os.path.join(DATASET_DIR, label)
    os.makedirs(folder, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_path = os.path.join(folder, filename + ".webm")
    wav_path = os.path.join(folder, filename + ".wav")

    with open(webm_path, "wb") as f:
        f.write(audio_data)

    subprocess.run(["ffmpeg", "-y", "-i", webm_path, wav_path])
    os.remove(webm_path)

    return jsonify({"success": True, "filename": filename + ".wav"})


if __name__ == "__main__":
    app.run(debug=True)

