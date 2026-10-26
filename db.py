"""SQLite storage layer. Schema is reloaded from the committed JSON file in
schemas/ on every startup (not stored long-term in the DB) since Streamlit
Cloud wipes local disk between restarts - only items/demo_fields get
re-synced idempotently; responses are expected to live for one working
session, same tradeoff as the original app.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "questionnaire_digitizer_ocr.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    position INTEGER NOT NULL,
    code TEXT NOT NULL,
    statement_en TEXT NOT NULL,
    statement_hi TEXT,
    UNIQUE(phase_id, position)
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    label TEXT,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    respondent_label TEXT,
    source_pdf_filename TEXT NOT NULL,
    pages_dir TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL REFERENCES responses(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    value INTEGER,
    confidence TEXT NOT NULL DEFAULT 'ok',
    note TEXT,
    source TEXT,
    y REAL,
    UNIQUE(response_id, item_id)
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def sync_schema_from_json(schema_path: Path) -> int:
    """Loads a schemas/*.json file and upserts its phase + items, returning
    the phase_id. Safe to call on every startup."""
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    conn = get_connection()
    try:
        conn.execute("INSERT OR IGNORE INTO phases (name) VALUES (?)", (schema["name"],))
        phase_id = conn.execute("SELECT id FROM phases WHERE name = ?", (schema["name"],)).fetchone()["id"]
        conn.executemany(
            "INSERT INTO items (phase_id, position, code, statement_en, statement_hi) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(phase_id, position) DO UPDATE SET "
            "code=excluded.code, statement_en=excluded.statement_en, statement_hi=excluded.statement_hi",
            [
                (phase_id, it["position"], it["code"], it["statement_en"], it.get("statement_hi"))
                for it in schema["items"]
            ],
        )
        conn.commit()
        return phase_id
    finally:
        conn.close()


def list_phases() -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute("SELECT id, name FROM phases ORDER BY id").fetchall()
    finally:
        conn.close()


def get_items(phase_id: int) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT id, position, code, statement_en, statement_hi FROM items "
            "WHERE phase_id = ? ORDER BY position",
            (phase_id,),
        ).fetchall()
    finally:
        conn.close()


def create_batch(phase_id: int, label: str | None = None) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO batches (phase_id, label, uploaded_at) VALUES (?, ?, ?)",
            (phase_id, label, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_batches(phase_id: int) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT id, label, uploaded_at FROM batches WHERE phase_id = ? ORDER BY id DESC",
            (phase_id,),
        ).fetchall()
    finally:
        conn.close()


def create_response(batch_id: int, source_pdf_filename: str, pages_dir: str,
                     respondent_label: str | None = None) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO responses (batch_id, respondent_label, source_pdf_filename, pages_dir) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, respondent_label, source_pdf_filename, pages_dir),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_responses(batch_id: int) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT id, respondent_label, source_pdf_filename, pages_dir FROM responses "
            "WHERE batch_id = ? ORDER BY id",
            (batch_id,),
        ).fetchall()
    finally:
        conn.close()


def get_response(response_id: int) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM responses WHERE id = ?", (response_id,)).fetchone()
    finally:
        conn.close()


def save_response_items(response_id: int, item_results: list) -> None:
    """item_results: list of ocr_pipeline.ItemResult."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM response_items WHERE response_id = ?", (response_id,))
        conn.executemany(
            "INSERT INTO response_items (response_id, item_id, value, confidence, note, source, y) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (response_id, r.item_id, r.value, r.confidence, r.note, r.source, r.y)
                for r in item_results
            ],
        )
        conn.commit()
    finally:
        conn.close()


def get_response_items(response_id: int) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT ri.id, ri.item_id, ri.value, ri.confidence, ri.note, ri.source, ri.y, "
            "       it.code, it.statement_en, it.position "
            "FROM response_items ri JOIN items it ON it.id = ri.item_id "
            "WHERE ri.response_id = ? ORDER BY it.position",
            (response_id,),
        ).fetchall()
    finally:
        conn.close()


def update_response_item(response_item_id: int, value: int, confidence: str = "ok") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE response_items SET value = ?, confidence = ? WHERE id = ?",
            (value, confidence, response_item_id),
        )
        conn.commit()
    finally:
        conn.close()
