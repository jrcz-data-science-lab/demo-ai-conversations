import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import base64
import tempfile
import requests
import time
import sys
import threading
import os

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
    try:
        sd.play(audio_np, samplerate=samplerate)
        sd.wait()
    except Exception as e:
        print("Playback error:", e)


def audio_to_base64(audio_np, samplerate=SAMPLERATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, samplerate, audio_np)
        with open(tmpfile.name, "rb") as f:
            audio_bytes = f.read()
    os.remove(tmpfile.name)
    return base64.b64encode(audio_bytes).decode("utf-8")


def base64_to_audio(audio_b64):
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        samplerate, data = read(tmpfile.name)
    os.remove(tmpfile.name)
    return data, samplerate


def generate_silence(duration=1.0, samplerate=SAMPLERATE):
    return np.zeros((int(duration * samplerate), 1), dtype=np.int16)


def send_audio(username, audio_b64, scenario, feedback=False):
    data = {
        "username": username,
        "audio": audio_b64,
        "feedback": feedback,
        "scenario": scenario
    }

    try:
        response = requests.post(SERVER_URL, json=data, timeout=500)
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

    scenario = input("Enter scenario number (e.g. 1, 2, 3...): ").strip()
    if not scenario.isdigit():
        print("Scenario must be a number.")
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
            audio_b64_resp = send_audio(username, audio_b64, scenario, feedback=False)

            if audio_b64_resp:
                print("Playing server response...")
                server_audio_np, sr = base64_to_audio(audio_b64_resp)
                play_audio(server_audio_np, sr)
            else:
                print("No audio received from server.")

            print("\n--- Ready for next round ---\n")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nConversation ended. Requesting feedback summary from server...")

        silent_audio = generate_silence()
        silent_b64 = audio_to_base64(silent_audio)

        # Send feedback request and get full response
        data = {
            "username": username,
            "audio": silent_b64,
            "feedback": True,
            "scenario": scenario
        }
        
        try:
            response = requests.post(SERVER_URL, json=data, timeout=60)
            response.raise_for_status()
            
            if response.headers.get("Content-Type", "").startswith("application/json"):
                feedback_json = response.json()
                audio_field = feedback_json.get("audio")
                print(feedback_json)
                
                if audio_field:
                    print("=== Feedback Summary ===")
                    try:
                        feedback_audio, sr = base64_to_audio(audio_field)
                        play_audio(feedback_audio, sr)
                    except Exception as e:
                        print(f"Error playing audio: {e}")
                    
                    # Display speech metrics and icon states if available
                    if "speech_metrics" in feedback_json:
                        print("\n=== Speech Metrics ===")
                        metrics = feedback_json["speech_metrics"]
                        print(f"Speech Rate (WPM): {metrics.get('speech_rate_wpm', 'N/A')}")
                        print(f"Average Pause: {metrics.get('avg_pause', 'N/A')}s")
                        print(f"Filler Count: {metrics.get('filler_count', 'N/A')}")
                        print(f"Filler Ratio: {metrics.get('filler_ratio', 'N/A')}%")
                        
                    if "icon_states" in feedback_json:
                        print("\n=== Icon States (for Unreal Engine) ===")
                        icons = feedback_json["icon_states"]
                        print(f"Speech Rate: {icons.get('speech_rate', 'N/A')}")
                        print(f"Pauses: {icons.get('pauses', 'N/A')}")
                        print(f"Fillers: {icons.get('fillers', 'N/A')}")
                        print(f"Overall: {icons.get('overall', 'N/A')}")
                    if "speech_summary" in feedback_json:
                        print(f"\n{feedback_json['speech_summary']}")
                    
                    # Display Gordon pattern analysis if available
                    if "gordon_patterns" in feedback_json:
                        print("\n=== Gordon Pattern Analysis ===")
                        patterns = feedback_json["gordon_patterns"]
                        print(f"Covered Patterns: {patterns.get('covered_patterns', 0)}/11")
                        print(f"Coverage Percentage: {patterns.get('coverage_percentage', 0)}%")
                        if patterns.get("mentioned_patterns"):
                            print(f"Mentioned Patterns: {patterns.get('mentioned_patterns', [])}")
                        if patterns.get("summary"):
                            print(f"\n{patterns.get('summary', '')}")
                else:
                    print("No audio in feedback response.")
                    print("Full response:", feedback_json)
            else:
                print("Unexpected response format:", response.text[:200])
        except Exception as e:
            print(f"Error requesting feedback: {e}")

        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
