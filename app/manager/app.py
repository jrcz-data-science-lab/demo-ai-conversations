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
        
        def on_open(ws):
            """Handle WebSocket opening"""
            print(f"[DEBUG] Whisper-live WebSocket opened for {session_id}")
            ws_ready.set()
            
            # Send initialization message to whisper-live
            init_msg = json.dumps({
                "uid": session_id,
                "language": language,
                "task": "transcribe",
                "model": "tiny",  # Use tiny for CPU performance
                "use_vad": True,
                "send_last_n_segments": 10,
                "no_speech_thresh": 0.45,
                "clip_audio": False,
                "same_output_threshold": 10
            })
            
            print(f"[DEBUG] Sending init message to whisper-live for {session_id}")
            try:
                ws.send(init_msg)
                print(f"[DEBUG] Init message sent successfully")
            except Exception as e:
                print(f"[ERROR] Failed to send init message: {e}")
        
        def on_message(ws, message):
            """Handle messages from whisper-live"""
            try:
                msg_data = json.loads(message)
                
                if 'status' in msg_data:
                    # Status messages
                    if msg_data['status'] == 'SERVER_READY':
                        print(f"Whisper-live ready for {session_id}")
                        socketio.emit('server_ready', {'session_id': session_id}, to=client_sid)
                    elif msg_data['status'] == 'ERROR':
                        socketio.emit('error', {'message': msg_data.get('message', 'Unknown error')}, to=client_sid)
                
                elif 'message' in msg_data:
                    # Message-based status (like SERVER_READY from faster_whisper)
                    if msg_data['message'] == 'SERVER_READY':
                        print(f"Whisper-live ready for {session_id}")
                        socketio.emit('server_ready', {'session_id': session_id}, to=client_sid)
                
                elif 'segments' in msg_data:
                    # Transcription segments received
                    segments = msg_data['segments']
                    full_text = ' '.join([seg['text'] for seg in segments if seg.get('completed', False)])
                    
                    # Store transcript
                    if session_id in whisper_connections:
                        whisper_connections[session_id]['full_transcript'] = full_text
                    
                    if full_text:
                        socketio.emit('transcription_update', {
                            'text': full_text,
                            'segments': segments
                        }, to=client_sid)
                
                elif 'language' in msg_data:
                    # Language detected
                    socketio.emit('language_detected', {
                        'language': msg_data['language'],
                        'probability': msg_data.get('language_prob', 0)
                    }, to=client_sid)
                    
            except json.JSONDecodeError:
                print(f"Failed to decode message from whisper-live: {message}")
            except Exception as e:
                print(f"Error processing whisper-live message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            socketio.emit('error', {'message': str(error)}, to=client_sid)
        
        def on_close(ws, close_status_code, close_msg):
            print(f"Whisper-live connection closed: {close_status_code}")
            if session_id in whisper_connections:
                del whisper_connections[session_id]
        
        # Create WebSocket connection to whisper-live
        ws = websocket.WebSocketApp(
            WHISPER_LIVE_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start connection in a thread
        def run_ws():
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        
        # Wait for connection to establish (max 5 seconds)
        ws_ready.wait(timeout=5)
        
        # Store connection
        whisper_connections[session_id] = {
            'ws': ws,
            'username': username,
            'scenario': scenario,
            'thread': ws_thread,
            'full_transcript': '',
            'client_sid': client_sid
        }
        
        emit('session_started', {'session_id': session_id})
        print(f"[DEBUG] Session {session_id} initialized successfully")
    
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
    
    try:
        # Decode base64 audio to float32 numpy array
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        
        # Send to whisper-live (binary mode)
        ws = whisper_connections[session_id]['ws']
        ws.send(audio_float32.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        emit('error', {'message': str(e)})

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

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)
