"""Database layer: sqlite3 connection factory + schema + repository.

# ponytail: sqlite via stdlib; the Postgres seam is connect() + Repo —
# swap for psycopg and %s placeholders when multi-tenant demands it.
"""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  message_id  TEXT PRIMARY KEY,
  received_at TEXT NOT NULL,            -- ISO 8601, normalised at ingest
  channel     TEXT NOT NULL,
  from_name   TEXT NOT NULL,
  body        TEXT NOT NULL,
  body_hash   TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(body_hash);

CREATE TABLE IF NOT EXISTS classifications (
  body_hash      TEXT PRIMARY KEY,      -- one LLM call per unique normalised body
  urgency        TEXT NOT NULL CHECK (urgency IN ('routine','same_day','emergency')),
  intent         TEXT NOT NULL,         -- JSON array as text
  score          INTEGER NOT NULL,      -- 0-100 absolute
  reason         TEXT NOT NULL,
  rule_hits      TEXT NOT NULL DEFAULT '[]',
  llm_urgency    TEXT NOT NULL,         -- pre-floor value: audits every rule escalation
  model          TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS corpus (     -- the retrieval pool = the adaptivity mechanism
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id TEXT,
  body       TEXT NOT NULL,
  body_hash  TEXT NOT NULL UNIQUE,
  urgency    TEXT NOT NULL CHECK (urgency IN ('routine','same_day','emergency')),
  dup_group  INTEGER NOT NULL,          -- near-duplicate cluster id; eval excludes own group
  source     TEXT NOT NULL CHECK (source IN ('seed','override')),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(db_path="app.db"):
    con = sqlite3.connect(db_path, check_same_thread=False)  # single-process demo; fastapi threadpool
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


class Repo:
    def __init__(self, con):
        self.con = con

    def upsert_message(self, m):
        self.con.execute(
            "INSERT OR IGNORE INTO messages (message_id, received_at, channel, from_name, body, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (m["message_id"], m["received_at"], m["channel"], m["from_name"], m["body"], m["body_hash"]),
        )
        self.con.commit()

    def get_message(self, message_id):
        row = self.con.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        return dict(row) if row else None

    def get_classification(self, body_hash):
        row = self.con.execute("SELECT * FROM classifications WHERE body_hash = ?", (body_hash,)).fetchone()
        return dict(row) if row else None

    def insert_classification(self, c):
        self.con.execute(
            "INSERT OR IGNORE INTO classifications "
            "(body_hash, urgency, intent, score, reason, rule_hits, llm_urgency, model, prompt_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c["body_hash"], c["urgency"], c["intent"], c["score"], c["reason"],
             c["rule_hits"], c["llm_urgency"], c["model"], c["prompt_version"]),
        )
        self.con.commit()

    def inbox(self):
        rows = self.con.execute(
            "SELECT m.message_id, m.received_at, m.channel, m.from_name, m.body, "
            "       c.urgency, c.score, c.reason "
            "FROM messages m JOIN classifications c ON c.body_hash = m.body_hash "
            "ORDER BY c.score DESC, m.received_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def corpus_rows(self):
        rows = self.con.execute("SELECT body, urgency, dup_group FROM corpus").fetchall()
        return [dict(r) for r in rows]

    def add_corpus_row(self, r):
        # an override on a body already in the pool replaces its label — the manager wins
        self.con.execute(
            "INSERT INTO corpus (message_id, body, body_hash, urgency, dup_group, source) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(body_hash) DO UPDATE SET urgency = excluded.urgency, source = excluded.source",
            (r["message_id"], r["body"], r["body_hash"], r["urgency"], r["dup_group"], r["source"]),
        )
        self.con.commit()

    def next_dup_group(self):
        return self.con.execute("SELECT COALESCE(MAX(dup_group), -1) + 1 AS g FROM corpus").fetchone()["g"]

    def corpus_size(self):
        return self.con.execute("SELECT COUNT(*) AS n FROM corpus").fetchone()["n"]
