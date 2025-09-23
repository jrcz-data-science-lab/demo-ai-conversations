from flask import Flask, request
import wave
from piper import PiperVoice
import os

app = Flask(__name__)

voice = PiperVoice.load("./nl_NL-ronnie-medium.onnx")

@app.post('/speech')
def speech():
    data = request.get_json()
    text = data.get("text")
    with wave.open("output.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
        os.system('aplay output.wav')
    return "ok"