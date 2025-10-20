import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import base64
import tempfile
import requests
import time
import sys
import threading

SERVER_URL = "http://145.19.54.110:8000/general"
SAMPLERATE = 48000
CHANNELS = 1

recording = []
is_recording = False


def callback(indata, frames, time_info, status):
    global recording
    if status:
        print(status)
    recording.append(indata.copy())


def record_audio_live():
    global recording, is_recording
    recording = []
    print("Recording... press ENTER again to stop.")
    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype="int16", callback=callback):
        while is_recording:
            sd.sleep(100)
    print("Recording stopped.")


def play_audio(audio_np, samplerate=SAMPLERATE):
    sd.play(audio_np, samplerate=samplerate)
    sd.wait()


def audio_to_base64(audio_np, samplerate=SAMPLERATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, samplerate, audio_np)
        with open(tmpfile.name, "rb") as f:
            audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def base64_to_audio(audio_b64):
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        samplerate, data = read(tmpfile.name)
    return data, samplerate


def send_audio(username, audio_b64):
    data = {"username": username, "audio": audio_b64, "feedback": False}

    try:
        response = requests.post(SERVER_URL, json=data, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            resp_json = response.json()
            if "error" in resp_json or (resp_json.get("audio") and str(resp_json.get("audio")).lower() == "error"):
                print("Server returned error:", resp_json)
                return None
        else:
            resp_text = response.text.strip() if response.text else ""
            if resp_text.lower() == "error":
                print("Server returned error:", resp_text)
                return None
            resp_json = {"audio": resp_text}

        audio_field = resp_json.get("audio")
        if audio_field is None or str(audio_field).lower() == "error":
            print("Server returned error:", resp_json)
            return None

        return audio_field

    except Exception as e:
        print("Error sending request:", e)
        return None


def main():
    global is_recording, recording

    username = input("Enter your username: ").strip()
    if not username:
        print("Username is required!")
        return

    print("\n=== Push-to-Talk Online Mode ===")
    print("Press ENTER to start recording, ENTER again to stop.")
    print("Audio will be sent to the server and the reply will play.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            input("Press ENTER to start recording...")
            is_recording = True
            rec_thread = threading.Thread(target=record_audio_live)
            rec_thread.start()

            input("")
            is_recording = False
            rec_thread.join()

            if not recording:
                print("No audio captured. Try again.")
                continue

            audio_np = np.concatenate(recording, axis=0)
            audio_b64 = audio_to_base64(audio_np)

            print("Sending audio to server...")
            audio_b64_resp = send_audio(username, audio_b64)

            if audio_b64_resp:
                print("Playing server response...")
                server_audio_np, sr = base64_to_audio(audio_b64_resp)
                play_audio(server_audio_np, sr)
            else:
                print("No audio received from server.")

            print("\n--- Ready for next round ---\n")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n Exiting gracefully. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
