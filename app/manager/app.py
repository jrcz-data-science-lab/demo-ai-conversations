from flask import Flask, request, jsonify
import requests
from db_utils import append_to_history, read_history
from sqlite import init_db
from user_management import ensure_user

app = Flask(__name__)

with open("prompts/conversation1.txt", "r", encoding="utf-8") as f:
    PROMPT_1 = f.read()

with open("prompts/feedback1.txt", "r", encoding="utf-8") as f:
    FEEDBACK_1 = f.read()

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'
STT_URL = 'http://faster-whisper:5000/transcribe'
GENERATE_URL = 'http://127.0.0.1:8000/generate'
FEEDBACK_URL = 'http://127.0.0.1:8000/feedback'

@app.route('/general', methods=['POST'])
def request_handling():
    data = request.json
    username = data.get("username")
    audio_in = data.get("audio")
    feedback_request = data.get("feedback", False)

    resp = requests.post(STT_URL, json={"audio": audio_in})
    transcription_text = resp.json().get("transcript", "")

    if not feedback_request:
        generate_resp = requests.post(GENERATE_URL, json={
            "username": username,
            "transcript": transcription_text
        })
        audio_b64 = generate_resp.json().get("audio")
        return jsonify({"audio": audio_b64})
    else:  
        feedback_resp = requests.post(FEEDBACK_URL, json={
            "username": username
        })
        audio_b64 = feedback_resp.json().get("audio")
        return jsonify({"audio": audio_b64})

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    username = data.get("username")
    transcript = data.get("transcript")

    if not username or not transcript:
        return jsonify({"error": "Missing username or transcript"}), 400

    ensure_user(username)
    append_to_history(username, "Student", transcript)
    convo = read_history(username)

    prompt_text = PROMPT_1.format(convo=convo)
    print( prompt_text)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt_text, "model": "mistral-small3.2:24b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        response_text = ollama_response.json().get("response", "")

        if response_text:
            append_to_history(username, "Avatar", response_text)
            tts_resp = requests.post(TTS_URL, json={"text": response_text})
            audio_b64 = tts_resp.json().get("audio")

        return jsonify({
        "response": response_text,
        "audio": audio_b64
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


@app.route('/feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "Missing username"}), 400

    ensure_user(username)
    convo = read_history(username)

    prompt_text = FEEDBACK_1.format(convo=convo)

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt_text, "model": "qwen3:32b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        feedback_text = ollama_response.json().get("response", "")

        if feedback_text:
            tts_resp = requests.post(TTS_URL, json={"text": feedback_text})
            audio_b64 = tts_resp.json().get("audio")

        return jsonify({
        "response": feedback_text,
        "audio": audio_b64
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8000)
