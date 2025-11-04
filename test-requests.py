import sounddevice as sd
import numpy as np
from scipy.io.wavfile import read
import base64
import tempfile
import socketio
import time
import sys
import threading
import os

SERVER_URL = "http://localhost:8000"
SAMPLERATE = 16000  # Whisper-live requires 16kHz
CHANNELS = 1
CHUNK_SIZE = 4096  # Audio chunk size for streaming

# Global state
is_streaming = False
session_id = None
last_transcription = ""  # Track last printed transcription to avoid duplicates

def audio_chunk_to_base64(audio_chunk):
    """Convert a single audio chunk to base64"""
    audio_bytes = audio_chunk.tobytes()
    return base64.b64encode(audio_bytes).decode("utf-8")


def base64_to_audio(audio_b64):
    """Convert base64 audio to numpy array"""
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        samplerate, data = read(tmpfile.name)
    os.remove(tmpfile.name)
    return data, samplerate


def play_audio(audio_np, samplerate):
    """Play audio using sounddevice"""
    try:
        sd.play(audio_np, samplerate=samplerate)
        sd.wait()
    except Exception as e:
        print("Playback error:", e)


def audio_callback(indata, frames, time_info, status):
    """Callback for continuous audio input"""
    global sio, session_id, is_streaming
    if status:
        print(f"Audio status: {status}")
    
    if is_streaming and session_id:
        # Convert audio chunk to base64 and send via WebSocket
        audio_chunk = indata.copy()
        audio_b64 = audio_chunk_to_base64(audio_chunk)
        try:
            sio.emit('audio_chunk', {
                'session_id': session_id,
                'audio': audio_b64
            })
        except Exception as e:
            print(f"Error sending audio chunk: {e}")


def main():
    global is_streaming, session_id, sio
    
    # Create Socket.IO client
    sio = socketio.Client()
    
    # Event handlers
    @sio.event
    def connect():
        print("Connected to server")
    
    @sio.event
    def disconnect():
        print("Disconnected from server")
    
    @sio.on('session_started')
    def on_session_started(data):
        global session_id
        session_id = data.get('session_id')
        print(f"Session started: {session_id}")
    
    @sio.on('server_ready')
    def on_server_ready(data):
        global is_streaming
        print("Whisper-live server is ready!")
        print("Start speaking... (speech will be detected automatically)")
        is_streaming = True
    
    @sio.on('transcription_update')
    def on_transcription_update(data):
        global last_transcription
        text = data.get('text', '')
        # Only print if text has changed
        if text and text != last_transcription:
            print(f"Transcription: {text}")
            last_transcription = text
    
    @sio.on('language_detected')
    def on_language_detected(data):
        lang = data.get('language', 'unknown')
        prob = data.get('probability', 0)
        print(f"Language detected: {lang} (confidence: {prob:.2f})")
    
    @sio.on('audio_response')
    def on_audio_response(data):
        global is_streaming
        audio_b64 = data.get('audio')
        if audio_b64:
            print("Playing server response...")
            try:
                audio_np, sr = base64_to_audio(audio_b64)
                play_audio(audio_np, sr)
                print("\n--- Ready for next round ---\n")
            except Exception as e:
                print(f"Error playing audio: {e}")
        is_streaming = False
    
    @sio.on('error')
    def on_error(data):
        message = data.get('message', 'Unknown error')
        print(f"Error: {message}")
    
    # Get user input
    username = input("Enter your username: ").strip()
    if not username:
        print("Username is required!")
        return
    
    scenario = input("Enter scenario number (e.g. 1, 2, 3...): ").strip()
    if not scenario.isdigit():
        print("Scenario must be a number.")
        return
    
    print("\n=== WebSocket Audio Streaming Mode ===")
    print("Voice activity detection enabled - speak naturally.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        # Connect to server
        sio.connect(SERVER_URL)
        time.sleep(0.5)
        
        # Start session
        sio.emit('start_session', {
            'username': username,
            'scenario': scenario,
            'language': 'nl'  # Dutch
        })
        time.sleep(1)  # Wait for session to initialize
        
        # Start continuous audio streaming
        print("Starting audio capture...")
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):
            # Keep the stream open until user interrupts
            while True:
                sd.sleep(100)
                
                # After a period of silence (VAD will detect), end the transcription
                # For now, we'll wait for user to manually end or for automatic detection
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nEnding session...")
        is_streaming = False
        
        if session_id and sio.connected:
            try:
                # Send end signal
                sio.emit('end_transcription', {'session_id': session_id})
                time.sleep(2)  # Wait for final response
            except Exception as e:
                print(f"Error ending session: {e}")
        
        if sio.connected:
            sio.disconnect()
        print("Goodbye!")
    
    except Exception as e:
        print(f"Error: {e}")
        sio.disconnect()


if __name__ == "__main__":
    main()