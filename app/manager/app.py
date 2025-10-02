import requests
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

OLLAMA_URL = 'http://ollama:11434/api/generate'
TTS_URL = 'http://tts:5000/speech'
DB_FILE = "chat.db"


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )""")
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        conn.commit()

def get_or_create_user(username):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        if row:
            return row[0]
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        return c.lastrowid

def append_to_history(username, speaker, text):
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (user_id, speaker, text) VALUES (?, ?, ?)",
            (user_id, speaker, text)
        )
        conn.commit()

def read_history(username):
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT speaker, text FROM messages WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,)
        )
        rows = c.fetchall()
    return "\n".join([f"{speaker}: {text}" for speaker, text in rows])


@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    username = data.get("username")
    transcript = data.get("transcript")

    if not username or not transcript:
        return jsonify({"error": "Missing username or transcript"}), 400

    append_to_history(username, "Student", transcript)
    convo = read_history(username)

    prompt = f"""
/nothink
Je speelt een AI-avatar in een game voor een HBO verpleegkunde opleiding.
De student verpleegkunde waar je zo mee praat, gaat leren hoe een diagnostisch gesprek gaat.
Het prompt dat je straks beantwoordt, komt uit een speech-to-text systeem en kan dus soms gekke woorden bevatten. Ik wil dat je je best doet om het eigenlijke woord te vinden.
Jouw output gaat door een text-to-speech systeem, dus gebruik geen rare symbolen of emoji.
Je moet spelen alsof je hoofdpijn hebt, en alsof de verpleegkundige een medische check bij je doet.
Hieronder volgt de historie van het gesprek. Als er "Student:" staat, heeft de student het gezegd. Als er "Avatar:" staat, heb jij het gezegd. Zorg ervoor dat er geen "Student:" of "Avatar:" in je output staat. Antwoord in maximaal 4 of 5 zinnen.

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
            tts_response = requests.post(TTS_URL, json={"text": response_text})
            tts_response.raise_for_status()

        return jsonify({"response": response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "Missing username"}), 400

    convo = read_history(username)

    prompt = f"""
/nothink
Je bent een docent of beoordelaar in een HBO verpleegkunde opleiding.
De student heeft net een diagnostisch gesprek gevoerd met een AI-avatar die een patiënt met hoofdpijn speelde.
Je taak is om inhoudelijke, vriendelijke en constructieve feedback te geven op de gespreksvaardigheden van de student, gebaseerd op de onderstaande conversatiegeschiedenis.

Let op de volgende aspecten:
- Stelt de student open en gerichte vragen?
- Toont de student empathie?
- Vat de student goed samen?
- Gaat de student in op de klachten van de patiënt?
- Gebruikt de student medische termen op een begrijpelijke manier?

Gebruik maximaal 5 zinnen.
Wees duidelijk en ondersteunend. Benoem wat goed gaat én wat beter kan.

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
            tts_response = requests.post(TTS_URL, json={"text": feedback_text})
            tts_response.raise_for_status()

        return jsonify({"feedback": feedback_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8000)
