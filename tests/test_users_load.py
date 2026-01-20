import os
import threading
import time
import random
import base64
import requests

# --- Configuration ---
SERVER_URL = "http://145.19.54.110:8000/general"
RECORDINGS_DIR = "test_recordings"  # folder with your interaction*.wav files
NUM_USERS = 8                       # number of simulated users
SCENARIO = 1                        # scenario number to send
MIN_DELAY = 6                       # min delay between interactions (s)
MAX_DELAY = 10                      # max delay between interactions (s)
REQUEST_TIMEOUT = 500               # timeout for requests (s)


# --- Helper function to convert WAV file to base64 ---
def wav_to_base64(path):
    """
    Reads a WAV file and returns a base64-encoded string of its bytes.
    This preserves the WAV header, just like the working client.
    """
    with open(path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


# --- Simulated user class ---
class SimulatedUser(threading.Thread):
    def __init__(self, username, audio_files):
        super().__init__(daemon=True)
        self.username = username
        self.audio_files = audio_files

    def run(self):
        print(f"[{self.username}] started")

        for wav_path in self.audio_files:
            # Random delay to simulate realistic speech intervals
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            time.sleep(delay)

            # Convert WAV to base64 string
            audio_b64 = wav_to_base64(wav_path)

            payload = {
                "username": self.username,
                "audio": audio_b64,
                "feedback": False,
                "scenario": SCENARIO
            }

            start_time = time.time()
            try:
                response = requests.post(
                    SERVER_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                latency = time.time() - start_time
                print(f"[{self.username}] sent {os.path.basename(wav_path)} ({latency:.2f}s)")
            except Exception as e:
                print(f"[{self.username}] ERROR sending {os.path.basename(wav_path)}: {e}")

        print(f"[{self.username}] finished")


# --- Main entry point ---
def main():
    # Get all WAV recordings in sorted order
    audio_files = sorted(
        os.path.join(RECORDINGS_DIR, f)
        for f in os.listdir(RECORDINGS_DIR)
        if f.lower().endswith(".wav")
    )

    if not audio_files:
        print("No recordings found in", RECORDINGS_DIR)
        return

    print(f"Loaded {len(audio_files)} recordings.")
    print(f"Starting load test with {NUM_USERS} users...\n")

    threads = []

    for i in range(NUM_USERS):
        user = SimulatedUser(
            username=f"user{i+1}",
            audio_files=audio_files
        )
        threads.append(user)
        user.start()

        # Slight stagger to avoid sending all requests at once
        time.sleep(random.uniform(0.2, 1.2))

    # Wait for all users to finish
    for t in threads:
        t.join()

    print("\n=== Load test completed ===")


if __name__ == "__main__":
    main()