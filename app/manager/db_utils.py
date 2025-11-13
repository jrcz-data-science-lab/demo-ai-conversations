import sqlite3
import json

DB_FILE = "chat.db"

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
        c.execute("INSERT INTO messages (user_id, speaker, text) VALUES (?, ?, ?)",
                  (user_id, speaker, text))
        conn.commit()
        return c.lastrowid  # Return the message_id

def read_history(username):
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT speaker, text FROM messages WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        rows = c.fetchall()
    return "\n".join([f"{speaker}: {text}" for speaker, text in rows])
    
def clear_history(username):
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        # Also clear audio metadata for privacy compliance
        c.execute("DELETE FROM conversation_audio_metadata WHERE user_id = ?", (user_id,))
        conn.commit()

def store_audio_metadata(username, message_id, audio_duration, transcript_details, word_count):
    """Store audio metadata for speech pattern analysis. Only metadata is stored, no audio recordings."""
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        # Convert transcript_details to JSON string if it's a dict
        transcript_details_json = json.dumps(transcript_details) if isinstance(transcript_details, dict) else transcript_details
        c.execute("""
            INSERT INTO conversation_audio_metadata 
            (user_id, message_id, audio_duration, transcript_details, word_count)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, message_id, audio_duration, transcript_details_json, word_count))
        conn.commit()

def get_all_audio_metadata(username):
    """Retrieve all audio metadata for a user's current session."""
    user_id = get_or_create_user(username)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT message_id, audio_duration, transcript_details, word_count, timestamp
            FROM conversation_audio_metadata
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        rows = c.fetchall()
    
    # Parse JSON transcript_details back to dict
    result = []
    for row in rows:
        message_id, audio_duration, transcript_details_json, word_count, timestamp = row
        try:
            transcript_details = json.loads(transcript_details_json) if transcript_details_json else {}
        except (json.JSONDecodeError, TypeError):
            transcript_details = {}
        
        result.append({
            "message_id": message_id,
            "audio_duration": audio_duration,
            "transcript_details": transcript_details,
            "word_count": word_count,
            "timestamp": timestamp
        })
    
    return result
