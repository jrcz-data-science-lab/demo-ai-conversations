from flask import Flask, request, jsonify
import requests
from db_utils import append_to_history, read_history, clear_history
from sqlite import init_db
from user_management import ensure_user
import os

app = Flask(__name__)

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'
STT_URL = 'http://faster-whisper:5000/transcribe'
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

@app.route('/general', methods=['POST'])
def request_handling():
    data = request.json
    username = data.get("username")
    audio_in = data.get("audio")
    scenario = data.get("scenario")
    feedback_request = data.get("feedback", False)

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

    if not username or not audio_in or not scenario:
        return jsonify({"error": "Missing username, audio, or scenario"}), 400

    # Transcribe audio
    stt_resp = requests.post(STT_URL, json={"audio": audio_in})
    transcription_text = stt_resp.json().get("transcript", "")

    if not feedback_request:
        generate_resp = requests.post(GENERATE_URL, json={
            "username": username,
            "transcript": transcription_text,
            "scenario": scenario,
            "voice": voice_model
        })
        audio_b64 = generate_resp.json().get("audio")
        return jsonify({"audio": audio_b64})
    else:
        feedback_resp = requests.post(FEEDBACK_URL, json={
            "username": username,
            "scenario": scenario,
            "voice": voice_model
        })
        audio_b64 = feedback_resp.json().get("audio")
        return jsonify({"audio": audio_b64})


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

        print(feedback_text)
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
    app.run(host='0.0.0.0', port=8000)
