from flask import Flask, request, jsonify
import requests
from db_utils import append_to_history, read_history
from sqlite import init_db
from user_management import ensure_user

app = Flask(__name__)

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'

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

    prompt = f"""
/nothink
Je speelt een AI-avatar in een game voor een HBO verpleegkunde opleiding.
De student voert een diagnostisch gesprek.
Hieronder volgt de gesprekshistorie:
{convo}
"""

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt, "model": "qwen3:0.6b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        response_text = ollama_response.json().get("response", "")

        if response_text:
            append_to_history(username, "Avatar", response_text)
            requests.post(TTS_URL, json={"text": response_text})

        return jsonify({"response": response_text})

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

    prompt = f"""
/nothink
Je bent een docent of beoordelaar in een HBO verpleegkunde opleiding.
De student heeft net een diagnostisch gesprek gevoerd met een AI-avatar.
Geef feedback in maximaal 5 zinnen, vriendelijk en constructief.

Gespreksgeschiedenis:
{convo}
"""

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={"prompt": prompt, "model": "qwen3:0.6b", "stream": False, "think": False}
        )
        ollama_response.raise_for_status()
        feedback_text = ollama_response.json().get("response", "")

        if feedback_text:
            requests.post(TTS_URL, json={"text": feedback_text})

        return jsonify({"feedback": feedback_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8000)
