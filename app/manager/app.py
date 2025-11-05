from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import requests
from db_utils import append_to_history, read_history, clear_history
from sqlite import init_db
from user_management import ensure_user
import os
import json
import threading
import websocket
import sys
import numpy as np
import base64
from io import BytesIO
import wave
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'
WHISPER_LIVE_URL = 'ws://faster-whisper-live:9090'
GENERATE_URL = 'http://127.0.0.1:8000/generate'
FEEDBACK_URL = 'http://127.0.0.1:8000/feedback'

def load_prompts(prompts_directory="prompts"):
    prompts = {}
    for filename in os.listdir(prompts_directory):
        if filename.endswith(".txt"):
            scenario_name = filename.split(".")[0]
            with open(os.path.join(prompts_directory, filename), "r", encoding="utf-8") as f:
                prompts[scenario_name] = f.read()
    return prompts

prompts = load_prompts() 

# Active WebSocket connections to whisper-live
whisper_connections = {}

# Track timestamps for inactivity detection per session
SILENCE_TIMEOUT = 3.0  # seconds of silence before triggering LLM
SILENCE_CHECK_INTERVAL = 0.5  # Check for silence every 0.5 seconds

# Silence checker class for background silence detection
class SilenceChecker:
    def __init__(self, session_id, check_interval=SILENCE_CHECK_INTERVAL):
        self.session_id = session_id
        self.check_interval = check_interval
        self.running = True
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(target=self._check_silence, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _check_silence(self):
        """Background thread that periodically checks for silence"""
        while self.running:
            time.sleep(self.check_interval)
            if not self.running:
                break
            
            if self.session_id not in whisper_connections:
                break
            
            conn = whisper_connections.get(self.session_id)
            if not conn:
                break
            
            last_time = conn.get('last_segment_time', time.time())
            time_since_last = time.time() - last_time
            
            if time_since_last > SILENCE_TIMEOUT:
                full_transcript = conn.get('full_transcript', '').strip()
                if full_transcript:
                    print(f"[DEBUG] [SILENCE] Silence detected for {self.session_id} ({time_since_last:.1f}s), triggering LLM")
                    # Reset timestamp to prevent repeated triggers
                    conn['last_segment_time'] = time.time()
                    process_full_transcription(self.session_id)

# WebSocket handler for audio streaming from frontend
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('start_session')
def handle_start_session(data):
    """Initialize a new transcription session with whisper-live"""
    try:
        from flask import request as flask_request
        import threading as th
        print(f"[DEBUG] start_session called with data: {data}")
        username = data.get('username')
        scenario = data.get('scenario')
        language = data.get('language', 'nl')  # Default to Dutch
        
        if not username or not scenario:
            emit('error', {'message': 'Missing username or scenario'})
            return
        
        # Create a WebSocket connection to whisper-live
        session_id = f"{username}_{scenario}"
        
        # Store client's socket.io session ID for later replies
        client_sid = flask_request.sid
        
        # Flag to track when connection is ready
        ws_ready = th.Event()
        server_ready = th.Event()  # Track when SERVER_READY is received
        
        def on_open(ws):
            """Handle WebSocket opening"""
            print(f"[DEBUG] [WS] WebSocket connection opened for {session_id}")
            ws_ready.set()
            
            # Send initialization message to whisper-live
            init_msg = json.dumps({
                "uid": session_id,
                "language": language,
                "task": "transcribe",
                "model": "large-v3-turbo",
                "use_vad": True,
                "send_last_n_segments": 10,
                "no_speech_thresh": 0.45,
                "clip_audio": False,
                "same_output_threshold": 10
            })
            
            print(f"[DEBUG] [WS] Sending init message to whisper-live for {session_id}")
            try:
                ws.send(init_msg)
                print(f"[DEBUG] [WS] Init message sent successfully to {WHISPER_LIVE_URL}")
            except Exception as e:
                print(f"[ERROR] [WS] Failed to send init message: {e}")
                socketio.emit('error', {'message': f"Failed to initialize whisper-live: {str(e)}"}, to=client_sid)
        
        def on_message(ws, message):
            """Handle messages from whisper-live"""
            try:
                msg_data = json.loads(message)
                
                if 'status' in msg_data:
                    # Status messages
                    if msg_data['status'] == 'SERVER_READY':
                        print(f"[DEBUG] [WS] SERVER_READY received for {session_id}")
                        server_ready.set()
                        socketio.emit('server_ready', {'session_id': session_id}, to=client_sid)
                    elif msg_data['status'] == 'ERROR':
                        error_msg = msg_data.get('message', 'Unknown error')
                        print(f"[ERROR] [WS] Error from whisper-live: {error_msg}")
                        socketio.emit('error', {'message': error_msg}, to=client_sid)
                
                elif 'message' in msg_data:
                    # Message-based status (like SERVER_READY from faster_whisper)
                    if msg_data['message'] == 'SERVER_READY':
                        print(f"[DEBUG] [WS] SERVER_READY (message format) received for {session_id}")
                        server_ready.set()
                        socketio.emit('server_ready', {'session_id': session_id}, to=client_sid)
                
                elif 'segments' in msg_data:
                    # Transcription segments received
                    segments = msg_data['segments']
                    full_text = ' '.join([seg['text'] for seg in segments if seg.get('completed', False)])
                    
                    print(f"[DEBUG] [TRANSCRIPT] Received segments for {session_id}: {full_text[:100]}...")
                    
                    # Store transcript and update last_segment_time
                    if session_id in whisper_connections:
                        whisper_connections[session_id]['full_transcript'] = full_text
                        whisper_connections[session_id]['last_segment_time'] = time.time()  # Update timestamp when segments received
                        print(f"[DEBUG] [TRANSCRIPT] Updated last_segment_time for {session_id}")
                    
                    if full_text:
                        socketio.emit('transcription_update', {
                            'text': full_text, 
                            'segments': segments
                        }, to=client_sid)

                elif 'language' in msg_data:
                    # Language detected
                    print(f"[DEBUG] [WS] Language detected for {session_id}: {msg_data['language']}")
                    socketio.emit('language_detected', {
                        'language': msg_data['language'],
                        'probability': msg_data.get('language_prob', 0)
                    }, to=client_sid)
                    
            except json.JSONDecodeError:
                print(f"[ERROR] [WS] Failed to decode message from whisper-live: {message[:100]}")
            except Exception as e:
                print(f"[ERROR] [WS] Error processing whisper-live message: {e}")
                import traceback
                traceback.print_exc()
        
        def on_error(ws, error):
            print(f"[ERROR] [WS] WebSocket error for {session_id}: {error}")
            socketio.emit('error', {'message': f"WebSocket error: {str(error)}"}, to=client_sid)
        
        def on_close(ws, close_status_code, close_msg):
            print(f"[DEBUG] [WS] Whisper-live connection closed for {session_id}: code={close_status_code}, msg={close_msg}")
            if session_id in whisper_connections:
                # Stop the silence checker thread
                if 'silence_checker' in whisper_connections[session_id]:
                    whisper_connections[session_id]['silence_checker'].stop()
                del whisper_connections[session_id]
        
        # Create WebSocket connection to whisper-live
        try:
            print(f"[DEBUG] [WS] Creating WebSocket connection to {WHISPER_LIVE_URL} for {session_id}")
            ws = websocket.WebSocketApp(
                WHISPER_LIVE_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start connection in a thread
            def run_ws():
                try:
                    ws.run_forever()
                except Exception as e:
                    print(f"[ERROR] [WS] WebSocket run_forever error for {session_id}: {e}")
                    socketio.emit('error', {'message': f"WebSocket connection error: {str(e)}"}, to=client_sid)
            
            ws_thread = threading.Thread(target=run_ws, daemon=True)
            ws_thread.start()
        except Exception as e:
            print(f"[ERROR] [WS] Failed to create WebSocket for {session_id}: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('error', {'message': f"Failed to create WebSocket connection: {str(e)}"}, to=client_sid)
            return
        
        # Wait for connection to establish (max 5 seconds)
        if not ws_ready.wait(timeout=5):
            print(f"[ERROR] [WS] WebSocket connection timeout for {session_id}")
            socketio.emit('error', {'message': 'Failed to connect to whisper-live server'}, to=client_sid)
            return
        
        # Wait for SERVER_READY (max 10 seconds)
        if not server_ready.wait(timeout=10):
            print(f"[WARNING] [WS] SERVER_READY not received for {session_id} within timeout")
        
        # Store connection with per-session silence tracking
        whisper_connections[session_id] = {
            'ws': ws,
            'username': username,
            'scenario': scenario,
            'thread': ws_thread,
            'full_transcript': '',
            'client_sid': client_sid,
            'last_segment_time': time.time(),  # Initialize timestamp
            'server_ready': server_ready.is_set()
        }
                
        silence_checker = SilenceChecker(session_id)
        silence_checker.start()
        whisper_connections[session_id]['silence_checker'] = silence_checker
        
        emit('session_started', {'session_id': session_id})
        print(f"[DEBUG] Session {session_id} initialized successfully (server_ready={server_ready.is_set()})")
    
    except Exception as e:
        print(f"[ERROR] Exception in handle_start_session: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f"Server error: {str(e)}"}, to=flask_request.sid if 'flask_request' in locals() else None)

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle audio chunk from client and forward to whisper-live"""
    session_id = data.get('session_id')
    audio_base64 = data.get('audio')
    
    if not session_id or not audio_base64:
        emit('error', {'message': 'Missing session_id or audio'})
        return
    
    if session_id not in whisper_connections:
        emit('error', {'message': 'Invalid session_id'})
        return
    
    conn = whisper_connections[session_id]
    
    # Check if server is ready
    if not conn.get('server_ready', False):
        # Silently drop audio chunks until server is ready
        return
    
    try:
        # Decode base64 audio to float32 numpy array
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Verify audio format: 16kHz, mono, int16 -> float32
        # Convert int16 [-32768, 32767] to float32 [-1.0, 1.0]
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        
        # Verify the audio is valid (not all zeros, reasonable size)
        if len(audio_float32) == 0:
            print(f"[WARNING] [AUDIO] Empty audio chunk received for {session_id}")
            return
        
        # Send to whisper-live (binary mode)
        # faster-whisper-live expects float32 binary data at 16kHz
        ws = conn['ws']
        audio_bytes_send = audio_float32.tobytes()
        ws.send(audio_bytes_send, opcode=websocket.ABNF.OPCODE_BINARY)
        
    except Exception as e:
        print(f"[ERROR] [AUDIO] Error processing audio chunk for {session_id}: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f"Audio processing error: {str(e)}"})

@socketio.on('end_transcription')
def handle_end_transcription(data):
    """End transcription and process with LLM"""
    from flask import request as flask_request
    session_id = data.get('session_id')
    
    if session_id not in whisper_connections:
        emit('error', {'message': 'Invalid session_id'})
        return
    
    conn = whisper_connections[session_id]
    username = conn['username']
    scenario = conn['scenario']
    ws = conn['ws']
    full_transcript = conn.get('full_transcript', '')
    client_sid = conn.get('client_sid')
    
    # Close whisper-live connection
    ws.send("END_OF_AUDIO")
    ws.close()
    del whisper_connections[session_id]
    
    if scenario == '1':
        voice_model = "Kumar Dahl"
    elif scenario == '2':
        voice_model = "Luis Moray"
    elif scenario == '3':
        voice_model = "Wulf Carlevaro"
    elif scenario == '4':
        voice_model = "Filip Traverse"
    else:
        voice_model = "Damien Black"
    
    # Process with LLM by calling the generate_response function directly
    try:
        ensure_user(username)
        append_to_history(username, "Student", full_transcript)
        convo = read_history(username)
        
        # Load the prompt
        prompt_text = prompts.get(f"conversation{scenario}", None)
        if not prompt_text:
            socketio.emit('error', {'message': f"No prompt found for scenario {scenario}"}, to=client_sid)
            return
        
        # Format the prompt with conversation history
        prompt_text = prompt_text.format(convo=convo)
        
        # Call Ollama
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt_text, "model": "mistral-small3.2:24b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        response_text = ollama_response.json().get("response", "")
        
        if response_text:
            append_to_history(username, "Avatar", response_text)
            
            # Call TTS
            tts_resp = requests.post(TTS_URL, json={"text": response_text, "voice": voice_model})
            audio_b64 = tts_resp.json().get("audio")
            
            socketio.emit('audio_response', {'audio': audio_b64}, to=client_sid)
        else:
            socketio.emit('error', {'message': "Empty response from model"}, to=client_sid)
            
    except Exception as e:
        socketio.emit('error', {'message': f"Error processing with LLM: {str(e)}"}, to=client_sid)

# Old REST endpoint - deprecated in favor of WebSocket
# @app.route('/general', methods=['POST'])
# def request_handling():
#     pass  # Now handled via WebSocket

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    username = data.get("username")
    transcript = data.get("transcript")
    scenario = data.get("scenario")
    voice = data.get("voice")

    if not username or not transcript or not scenario:
        return jsonify({"error": "Missing username, transcript, or scenario"}), 400

    ensure_user(username)
    append_to_history(username, "Student", transcript)
    convo = read_history(username)

    # Dynamically load the prompt from the dictionary
    prompt_text = prompts.get(f"conversation{scenario}", None)
    if not prompt_text:
        return jsonify({"error": f"No prompt found for scenario {scenario}"}), 400

    # Format the prompt with the conversation history
    prompt_text = prompt_text.format(convo=convo)
    print(prompt_text)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt_text, "model": "mistral-small3.2:24b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        response_text = ollama_response.json().get("response", "")
        print(response_text)

        if response_text:
            append_to_history(username, "Avatar", response_text)
            tts_resp = requests.post(TTS_URL, json={"text": response_text, "voice": voice})
            audio_b64 = tts_resp.json().get("audio")
            return jsonify({"response": response_text, "audio": audio_b64})

        return jsonify({"error": "Empty response from model"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama error: {e}"}), 500

@app.route('/feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    username = data.get("username")
    scenario = data.get("scenario")
    voice = data.get("voice")

    if not username or not scenario:
        return jsonify({"error": "Missing username or scenario"}), 400

    ensure_user(username)
    convo = read_history(username)

    # Dynamically load the feedback prompt from the dictionary
    feedback_text = prompts.get(f"feedback{scenario}", None)
    if not feedback_text:
        return jsonify({"error": f"No feedback prompt for scenario {scenario}"}), 400

    # Format the prompt with the conversation history
    feedback_text = feedback_text.format(convo=convo)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": feedback_text, "model": "qwen3:32b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        feedback_text = ollama_response.json().get("response", "")

        if feedback_text:
            tts_resp = requests.post(TTS_URL, json={"text": feedback_text, "voice": voice})
            audio_b64 = tts_resp.json().get("audio")

            clear_history(username)

            return jsonify({"response": feedback_text, "audio": audio_b64})

        return jsonify({"error": "Empty feedback response"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama error: {e}"}), 500

def process_full_transcription(session_id):
    conn = whisper_connections.get(session_id)
    if not conn:
        print(f"[WARNING] [LLM] No connection found for {session_id}")
        return

    username = conn['username']
    scenario = conn['scenario']
    client_sid = conn['client_sid']
    full_transcript = conn.get('full_transcript', '').strip()

    if not full_transcript:
        print(f"[DEBUG] [LLM] No text to process yet for {session_id}")
        return
    
    print(f"[DEBUG] [LLM] Processing transcription for {session_id}: '{full_transcript[:100]}...'")

    # Prepare the request payload as expected by /generate
    data = {
        "username": username,
        "transcript": full_transcript,
        "scenario": scenario,
        "voice": "Filip Traverse"  # or any dynamic choice
    }

    try:
        # Make the HTTP request to /generate route
        response = requests.post(GENERATE_URL, json=data)

        # Check for errors in the response
        if response.status_code != 200:
            raise Exception(f"Error generating response: {response.text}")

        # Process the response from /generate
        result = response.json()
        if "error" in result:
            raise Exception(f"Error: {result['error']}")

        response_text = result["response"]
        audio_b64 = result["audio"]

        # Emit response back to the client
        socketio.emit('audio_response', {'audio': audio_b64, 'text': response_text}, to=client_sid)

        print(f"[DEBUG] Sent LLM + TTS response to {session_id}")

        # Optionally clear transcript for next message
        whisper_connections[session_id]['full_transcript'] = ""

    except Exception as e:
        print(f"[ERROR] LLM processing failed: {e}")
        socketio.emit('error', {'message': str(e)}, to=client_sid)

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)