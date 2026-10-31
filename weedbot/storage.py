"""SQLite persistence for human review labels and planner proposals."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    image_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    planner_name TEXT NOT NULL,
    planner_config_json TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    valid_weed_target TEXT NOT NULL,
    approach_safety TEXT NOT NULL,
    weed_removed TEXT NOT NULL,
    crop_damaged TEXT NOT NULL,
    damage_severity TEXT NOT NULL,
    annotator_confidence TEXT NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_image_id ON feedback(image_id);
CREATE INDEX IF NOT EXISTS idx_feedback_proposal_id ON feedback(proposal_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    connection.commit()
    return connection


def save_feedback(db_path: str | Path, payload: dict[str, Any]) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    connection = connect(db_path)
    try:
        cursor = connection.execute(
            """INSERT INTO feedback (
                created_at, image_id, image_path, proposal_id, planner_name,
                planner_config_json, proposal_json, valid_weed_target, approach_safety,
                weed_removed, crop_damaged, damage_severity, annotator_confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_at,
                payload["image_id"],
                payload["image_path"],
                payload["proposal_id"],
                payload["planner_name"],
                json.dumps(payload["planner_config"], sort_keys=True),
                json.dumps(payload["proposal"], sort_keys=True),
                payload["valid_weed_target"],
                payload["approach_safety"],
                payload["weed_removed"],
                payload["crop_damaged"],
                payload["damage_severity"],
                payload["annotator_confidence"],
                payload.get("notes", ""),
            ),
        )
        connection.commit()
        feedback_id = int(cursor.lastrowid)
    finally:
        connection.close()
    return feedback_id


def export_feedback_csv(db_path: str | Path, csv_path: str | Path) -> int:
    import csv

    db_path, csv_path = Path(db_path), Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        rows = connection.execute("SELECT * FROM feedback ORDER BY feedback_id").fetchall()
        headers = [description[0] for description in connection.execute("SELECT * FROM feedback LIMIT 0").description]
    finally:
        connection.close()
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    return len(rows)
