"""SQLite persistence for human review labels and planner proposals.

Schema v2: the detailed multi-question feedback form is replaced by a single
``pull_decision_correct`` field (correct / incorrect / uncertain).

A migration runs automatically on first connect so existing databases created
with the old schema (v1) are upgraded in-place without data loss where
possible.  Old-schema rows are preserved with ``pull_decision_correct`` set to
``NULL`` (stored as an empty string since the column is NOT NULL after rebuild).
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              TEXT    NOT NULL,
    image_id                TEXT    NOT NULL,
    image_path              TEXT    NOT NULL,
    proposal_id             TEXT    NOT NULL,
    planner_name            TEXT    NOT NULL,
    planner_config_json     TEXT    NOT NULL,
    proposal_json           TEXT    NOT NULL,
    pull_decision_correct   TEXT    NOT NULL,
    notes                   TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_image_id    ON feedback(image_id);
CREATE INDEX IF NOT EXISTS idx_feedback_proposal_id ON feedback(proposal_id);
"""

# Columns present in the old v1 schema that no longer exist in v2.
_V1_ONLY_COLUMNS = {
    "valid_weed_target",
    "approach_safety",
    "weed_removed",
    "crop_damaged",
    "damage_severity",
    "annotator_confidence",
}


# ---------------------------------------------------------------------------
# Connection + migration
# ---------------------------------------------------------------------------

def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _migrate(conn: sqlite3.Connection) -> None:
    """Migrate an existing database to the v2 schema if necessary."""
    # If the table doesn't exist yet, executescript will create it.
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "feedback" not in tables:
        conn.executescript(_SCHEMA_V2)
        conn.commit()
        return

    existing = _column_names(conn, "feedback")

    # Already on v2 — nothing to do.
    if "pull_decision_correct" in existing and not existing.intersection(_V1_ONLY_COLUMNS):
        return

    # Need to migrate: rebuild the table.
    # 1. Rename old table.
    conn.execute("ALTER TABLE feedback RENAME TO feedback_v1_backup")
    # 2. Create new table.
    conn.executescript(_SCHEMA_V2)

    # 3. Copy rows that exist in both schemas, filling in defaults.
    old_cols = _column_names(conn, "feedback_v1_backup")
    shared = [
        c for c in (
            "feedback_id", "created_at", "image_id", "image_path",
            "proposal_id", "planner_name", "planner_config_json",
            "proposal_json", "notes",
        )
        if c in old_cols
    ]

    if shared:
        col_list = ", ".join(shared)
        # pull_decision_correct is new — use empty string (unknown) for old rows.
        conn.execute(
            f"INSERT INTO feedback ({col_list}, pull_decision_correct) "
            f"SELECT {col_list}, '' FROM feedback_v1_backup"
        )

    # 4. Drop old backup table.
    conn.execute("DROP TABLE IF EXISTS feedback_v1_backup")
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_feedback(db_path: str | Path, payload: dict[str, Any]) -> int:
    """Persist one feedback record; return the new ``feedback_id``."""
    created_at = datetime.now(timezone.utc).isoformat()
    conn = connect(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO feedback (
                created_at, image_id, image_path, proposal_id, planner_name,
                planner_config_json, proposal_json, pull_decision_correct, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_at,
                payload["image_id"],
                payload["image_path"],
                payload["proposal_id"],
                payload["planner_name"],
                json.dumps(payload.get("planner_config", {}), sort_keys=True),
                json.dumps(payload.get("proposal", {}), sort_keys=True),
                payload["pull_decision_correct"],
                payload.get("notes", "") or "",
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def count_labeled_feedback(db_path: str | Path) -> int:
    """Return the number of feedback rows with a non-empty pull_decision_correct."""
    try:
        conn = connect(db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE pull_decision_correct IN ('correct','incorrect')"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def export_feedback_csv(db_path: str | Path, csv_path: str | Path) -> int:
    """Export all feedback rows to *csv_path*; return the row count."""
    db_path, csv_path = Path(db_path), Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM feedback ORDER BY feedback_id").fetchall()
        headers = [
            d[0]
            for d in conn.execute("SELECT * FROM feedback LIMIT 0").description
        ]
    finally:
        conn.close()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    return len(rows)
