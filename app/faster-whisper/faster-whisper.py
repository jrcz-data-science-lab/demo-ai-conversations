import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
from faster_whisper import WhisperModel
import os
import requests

print(sd.query_devices())

model = WhisperModel("tiny", compute_type="int8")

SAMPLERATE = 48000
chunk_duration = 5.0
CHUNK_SAMPLES = int(SAMPLERATE * chunk_duration)

api_url = "http://manager:8000/generate"
feedback_url = "http://manager:8000/feedback"

sd.default.device = (4, None)

counter = 0

USERNAME = "Dean"

def record_chunk():
    print("Recording chunk...")
    audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLERATE, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

def transcribe_chunk(audio_np):
    print('Transcribing...')
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, SAMPLERATE, audio_np)
        segments, _ = model.transcribe(tmpfile.name, language="nl", vad_filter=True)
        os.unlink(tmpfile.name)

        transcript = " ".join([seg.text for seg in segments]).strip()
        print(f"[Transcript]: {transcript}")
        return transcript

def main():
    global counter

    print("Starting transcription in chunks...")
    while True:
        audio_chunk = record_chunk()
        text = transcribe_chunk(audio_chunk)

        if text:
            counter += 1
            payload = {
                "username": USERNAME,
                "transcript": text
            }

            if counter % 5 == 0:
                try:
                    requests.post(feedback_url, json=payload)
                    print(f"[INFO] Feedback requested via /feedback (count {counter})")
                except Exception as e:
                    print(f"[ERROR] Failed to send to /feedback: {e}")
            else:
                try:
                    requests.post(api_url, json=payload)
                    print(f"[INFO] Transcript sent to /generate (count {counter})")
                except Exception as e:
                    print(f"[ERROR] Failed to send to /generate: {e}")

if __name__ == "__main__":
    main()
