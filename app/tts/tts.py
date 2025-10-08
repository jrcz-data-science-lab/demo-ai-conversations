from flask import Flask, request, jsonify
import wave
from piper import PiperVoice
import io
import base64

app = Flask(__name__)

voice = PiperVoice.load("./nl_NL-ronnie-medium.onnx")

@app.post('/speech')
def speech():
    data = request.get_json()
    text = data.get("text")

    # Write audio to memory buffer instead of disk
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        voice.synthesize_wav(text, wav_file)

    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return jsonify({"status": "ok", "audio": audio_b64})
