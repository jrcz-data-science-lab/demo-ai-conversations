import sqlite3

DB_FILE = "chat.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        # This table stores only session-level metadata (durations, timestamps)
        # NO audio recordings or transcripts of private content are stored permanently.
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_audio_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            audio_duration REAL,
            transcript_details TEXT,
            word_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(message_id) REFERENCES messages(id)
        )
        """)
        conn.commit()
