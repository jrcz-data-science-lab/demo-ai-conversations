import sqlite3

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
        conn.commit()
