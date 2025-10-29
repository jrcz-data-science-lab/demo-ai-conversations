from flask import Flask, request, jsonify
import torch
from TTS.api import TTS
import io
import base64
import tempfile
import os
import sys
from io import StringIO

def simulate_input(input_text):
    sys.stdin = StringIO(input_text)

simulate_input("y\n")

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
speaker = "Wulf Carlevaro"
# Kumar Dahl
# Luis Moray
# Wulf Carlevaro
# Filip Traverse
# Damien Black
 
@app.post('/speech')
def speech():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmp_path = tmpfile.name

    try:
        # Generate speech and save to the temporary file
        tts.tts_to_file(text=text, speaker=speaker, language="nl", file_path=tmp_path)

        # Read the audio back into memory
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        # Encode to base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return jsonify({"status": "ok", "audio": audio_b64})

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

