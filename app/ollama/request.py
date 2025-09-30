import requests
from flask import Flask, request, jsonify
from pathlib import Path

app = Flask(__name__)

OLLAMA_URL = 'http://145.19.54.111:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'

HISTORY_FILE = Path("chat_history.txt")

def append_to_history(speaker, text):
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{speaker}: {text}\n")

def read_history():
    """Return full conversation so far (or empty string)."""
    if HISTORY_FILE.exists():
        return HISTORY_FILE.read_text(encoding="utf-8")
    return ""

@app.route('/generate', methods=['POST'])
def generate_response():
    transcript = request.json.get("transcript")

    append_to_history("Student", transcript)
    convo = read_history()

    prompt = f"""
            /nothink
            /no_think
            Je speelt een AI-avatar in een game voor een HBO verpleegkunde opleiding.
            De student verpleegkunde waar je zo mee praat, gaat leren hoe een diagnostisch gesprek gaat.
            Het prompt dat je straks beantwoordt, komt uit een speech-to-text systeem en kan dus soms gekke woorden bevatten. Ik wil dat je je best doet om het eigenlijke woord te vinden.
            Jouw output gaat door een text-to-speech systeem, dus gebruik geen rare symbolen of emoji.
            Je moet spelen alsof je hoofdpijn hebt, en alsof de verpleegkundige een medische check bij je doet.
            Hieronder volgt de historie van het gesprek. Als er "Student:" staat, heeft de student het gezegd. Als er "Avatar: " staat, heb jij het gezegd. Zorg ervoor dat er geen "Student: " of "Avatar: " in je output staat. Antwoord in maximaal 4 of 5 zinnen.
            {convo}
            """

    try:
        ollama_response = requests.post(OLLAMA_URL, json={"prompt": prompt, "model": "gemma3:27b", "stream": False, "think": False})
        ollama_response.raise_for_status()
        
        response_text = ollama_response.json().get("response", "")
        append_to_history("Avatar", response_text)

        if response_text:
            tts_response = requests.post(TTS_URL, json={"text": response_text})
            tts_response.raise_for_status()

        return jsonify({"response": response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
