import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'

@app.route('/generate', methods=['POST'])
def generate_response():
    prompt = request.json.get("prompt")

    try:
        ollama_response = requests.post(OLLAMA_URL, json={"prompt": prompt, "model": "qwen3:8b", "stream": False, "think": False})
        ollama_response.raise_for_status()
        
        response_text = ollama_response.json().get("response", "")

        if response_text:
            tts_response = requests.post(TTS_URL, json={"text": response_text})
            tts_response.raise_for_status()

        return jsonify({"response": response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
