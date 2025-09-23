import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
from faster_whisper import WhisperModel
import os
import requests

model = WhisperModel("large", compute_type="int8")

SAMPLERATE = 48000
chunk_duration = 5.0
CHUNK_SAMPLES = int(SAMPLERATE * chunk_duration)
api_url = "http://ollama:8000/generate"
sd.default.device = (4, None)

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

    return " ".join([seg.text for seg in segments]).strip()
    
def main():
    print("Starting transcription in chunks...")
    while True:
        audio_chunk = record_chunk()
        text = transcribe_chunk(audio_chunk)
        if text:
            prompt = f"Antwoord in maximaal 2 zinnen: Prompt: {text}"

            requests.post(api_url, json={"prompt": prompt})
            print(f"[Transcript]: {text}")

if __name__ == "__main__":
    main()
