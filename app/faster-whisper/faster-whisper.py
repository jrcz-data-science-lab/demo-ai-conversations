from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import base64
import io
import tempfile
import json

app = Flask(__name__)

# Change Model Type
model = WhisperModel("large-v3-turbo", compute_type="int8")

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
            tmp.flush()

            # Transcribe using faster whisper
            segments, info = model.transcribe(tmp.name, language="nl", vad_filter=True)
            text = " ".join([segment.text for segment in segments])
            
            # Extract timestamped segments for pause detection
            transcript_details = {
                "segments": [
                    {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    }
                    for segment in segments
                ],
                "audio_duration": info.duration if hasattr(info, 'duration') else None,
                "word_timestamps": None  # faster-whisper segments don't have word-level timestamps by default
            }
            
            # Calculate word count for fallback
            word_count = len(text.split())
            
            # Calculate audio duration from segments if not available
            if transcript_details["audio_duration"] is None and transcript_details["segments"]:
                last_segment = transcript_details["segments"][-1]
                transcript_details["audio_duration"] = last_segment.get("end", 0)
        
        # Fallback handling: if segments are empty or timestamps missing, provide basic info
        if not transcript_details["segments"] or transcript_details["audio_duration"] is None:
            # Estimate duration from audio file size (rough approximation)
            # For fallback, we'll return minimal data
            transcript_details["audio_duration"] = transcript_details.get("audio_duration") or 0
            transcript_details["word_count"] = word_count

        return jsonify({
            "transcript": text.strip(),
            "transcript_details": transcript_details
        })

    except Exception as e:
        print({"error": f"Transcription failed: {e} Details: {str(e)}"})
        # Fallback: return basic error response with word count if available
        try:
            # Try to get word count even on error
            text = " ".join([segment.text for segment in segments]) if 'segments' in locals() else ""
            word_count = len(text.split()) if text else 0
            return jsonify({
                "error": f"Transcription failed: {e}",
                "transcript": text.strip() if text else "",
                "transcript_details": {
                    "audio_duration": 0,
                    "word_count": word_count,
                    "segments": []
                }
            }), 500
        except:
            return jsonify({"error": f"Transcription failed: {e} Details: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)