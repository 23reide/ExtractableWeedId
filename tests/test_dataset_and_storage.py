"""Tests for dataset loading and storage (updated for v2 binary feedback schema)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from weedbot.dataset import load_scenes
from weedbot.storage import export_feedback_csv, save_feedback, count_labeled_feedback

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_payload(**overrides) -> dict:
    base = {
        "image_id": "unit_test_image",
        "image_path": "/some/image.png",
        "proposal_id": "unit_test_image::weed_1::bbox_clearance_v1",
        "planner_name": "bbox_clearance_v1",
        "planner_config": {"gripper_radius_px": 8},
        "proposal": {"decision": "attempt", "target_xy": [3, 4]},
        "pull_decision_correct": "correct",
        "notes": "unit test",
    }
    base.update(overrides)
    return base


class DatasetTests(unittest.TestCase):
    def test_load_synthetic_dataset(self):
        scenes = load_scenes(PROJECT_ROOT / "demo_dataset", "CropsOrWeed9")
        self.assertEqual(len(scenes), 2)
        self.assertEqual(len(scenes[0].weeds), 2)
        self.assertEqual(len(scenes[0].crops), 2)
        self.assertTrue(
            all(0 <= weed.base_xy[0] < scenes[0].image_width for weed in scenes[0].weeds)
        )


class StorageV2Tests(unittest.TestCase):
    def test_feedback_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            fid = save_feedback(db, _make_payload())
            self.assertEqual(fid, 1)

    def test_pull_decision_correct_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            save_feedback(db, _make_payload(pull_decision_correct="correct"))
            save_feedback(db, _make_payload(pull_decision_correct="incorrect"))
            save_feedback(db, _make_payload(pull_decision_correct="uncertain"))
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT pull_decision_correct FROM feedback ORDER BY feedback_id"
            ).fetchall()
            conn.close()
            self.assertEqual([r[0] for r in rows], ["correct", "incorrect", "uncertain"])

    def test_count_labeled_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            save_feedback(db, _make_payload(pull_decision_correct="correct"))
            save_feedback(db, _make_payload(pull_decision_correct="incorrect"))
            save_feedback(db, _make_payload(pull_decision_correct="uncertain"))
            self.assertEqual(count_labeled_feedback(db), 2)

    def test_export_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            csv_path = Path(tmp) / "out.csv"
            save_feedback(db, _make_payload())
            rows = export_feedback_csv(db, csv_path)
            self.assertEqual(rows, 1)
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("unit_test_image", content)
            self.assertIn("unit test", content)
            self.assertIn("pull_decision_correct", content)

    def test_old_schema_migration(self):
        """Connecting to a v1 database should migrate it without error."""
        v1_schema = """
        CREATE TABLE feedback (
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
        );"""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.sqlite"
            # Create a v1-schema database with one row.
            conn = sqlite3.connect(db)
            conn.executescript(v1_schema)
            conn.execute(
                "INSERT INTO feedback VALUES (1,'2024-01-01','img1','/img.png','p1',"
                "'planner','{}','{}','yes','safe','likely_yes','no','minor','high','note')"
            )
            conn.commit()
            conn.close()

            # Now connect through the storage module — should migrate cleanly.
            from weedbot.storage import connect as _connect
            conn2 = _connect(db)
            cols = {r[1] for r in conn2.execute("PRAGMA table_info(feedback)").fetchall()}
            conn2.close()

            self.assertIn("pull_decision_correct", cols)
            self.assertNotIn("valid_weed_target", cols)


if __name__ == "__main__":
    unittest.main()
