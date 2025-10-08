from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import base64
import io
import tempfile

app = Flask(__name__)

model = WhisperModel("tiny", compute_type="int8")

@app.post('/transcribe')
def speech():
    data = request.get_json()
    audio_base64 = data.get("audio")

    if not audio_base64:
        return jsonify({"error": "Missing 'audio' field"}), 400

    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return jsonify({"error": f"Invalid base64 audio: {e}"}), 400

    try:
        # write audio to wav file
        with tempfile.NamedTemporaryFile(suffix="wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush

            # Transcribe using faster whisper
            segments, info = model.transcribe(tmp.name, language="nl", vad_filter=True)
            text = " ".join([segment.text for segment in segments])

        return jsonify({"transcript": text.strip()})

    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)