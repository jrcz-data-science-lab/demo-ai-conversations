import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import base64
import tempfile
import requests
import io

SERVER_URL = "http://145.19.54.110:8000/general"
SAMPLERATE = 48000
DURATION = 5
CHANNELS = 1


def record_audio(duration=DURATION, samplerate=SAMPLERATE, channels=CHANNELS):
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    return np.squeeze(audio)


def play_audio(audio_np, samplerate=SAMPLERATE):
    print("Playing audio...")
    sd.play(audio_np, samplerate=samplerate)
    sd.wait()
    print("Playback finished.")


def audio_to_base64(audio_np, samplerate=SAMPLERATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, samplerate, audio_np)
        with open(tmpfile.name, "rb") as f:
            audio_bytes = f.read()
    encoded = base64.b64encode(audio_bytes).decode('utf-8')
    return encoded


def base64_to_wav_file(audio_b64):
    audio_bytes = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.flush()
    return tmp.name


def main():
    username = input("Enter your username: ").strip()
    if not username:
        print("Username is required!")
        return

    audio_np = record_audio()
    play_audio(audio_np)

    audio_b64 = audio_to_base64(audio_np)
    data = {
        "username": username,
        "audio": audio_b64,
        "feedback": False
    }

    try:
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status()
        resp_json = response.json()

        if "audio" in resp_json:
            print("Received audio from server — decoding and playing...")
            audio_b64_resp = resp_json["audio"]

            wav_path = base64_to_wav_file(audio_b64_resp)
            samplerate, audio_np = read(wav_path)

            play_audio(audio_np, samplerate)
        else:
            print("No audio found in response:", resp_json)

    except requests.exceptions.RequestException as e:
        print("Error sending request:", e)


if __name__ == "__main__":
    main()
