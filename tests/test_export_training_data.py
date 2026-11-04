"""Tests for the training data export pipeline."""
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from weedbot.export_training_data import export_for_training, zip_training_export
from weedbot.storage import save_feedback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_IMAGE = PROJECT_ROOT / "demo_dataset" / "images" / "demo_0001.png"


def _make_payload(pull_decision_correct: str, proposal_id: str = "img::weed::planner") -> dict:
    return {
        "image_id": "demo_0001",
        "image_path": str(DEMO_IMAGE),
        "proposal_id": proposal_id,
        "planner_name": "bbox_clearance_v1",
        "planner_config": {"gripper_radius_px": 8},
        "proposal": {
            "decision": "attempt",
            "target_xy": [160, 157],  # valid coords within demo image
        },
        "pull_decision_correct": pull_decision_correct,
        "notes": "",
    }


@unittest.skipUnless(DEMO_IMAGE.exists(), "demo dataset not generated — run make_demo_data.py")
class ExportTrainingDataTests(unittest.TestCase):
    def test_export_skips_uncertain(self):
        """Only 'correct'/'incorrect' rows should be exported; 'uncertain' is excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            save_feedback(db, _make_payload("correct",   "p1"))
            save_feedback(db, _make_payload("incorrect", "p2"))
            save_feedback(db, _make_payload("uncertain", "p3"))
            count = export_for_training(db, out)
            self.assertEqual(count, 2)

    def test_export_creates_patches_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            save_feedback(db, _make_payload("correct",   "p1"))
            save_feedback(db, _make_payload("incorrect", "p2"))
            export_for_training(db, out)

            images_dir = out / "images"
            self.assertTrue(images_dir.exists())
            patches = list(images_dir.glob("*.png"))
            self.assertEqual(len(patches), 2)

            labels_csv = out / "labels.csv"
            self.assertTrue(labels_csv.exists())
            with labels_csv.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)

    def test_patch_is_correct_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            save_feedback(db, _make_payload("correct", "p1"))
            export_for_training(db, out, patch_size=128)
            patches = list((out / "images").glob("*.png"))
            with Image.open(patches[0]) as img:
                self.assertEqual(img.size, (128, 128))

    def test_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            save_feedback(db, _make_payload("correct", "p1"))
            export_for_training(db, out)
            with (out / "labels.csv").open() as fh:
                reader = csv.DictReader(fh)
                expected = {
                    "patch_path", "image_id", "proposal_id",
                    "model_decision", "pull_decision_correct", "annotated_at",
                }
                self.assertEqual(expected, set(reader.fieldnames or []))

    def test_retrain_log_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            save_feedback(db, _make_payload("correct", "p1"))
            export_for_training(db, out)
            log_path = out / "retrain_log.json"
            self.assertTrue(log_path.exists())
            log = json.loads(log_path.read_text())
            self.assertEqual(log["sample_count"], 1)

    def test_zip_contains_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fb.sqlite"
            out = Path(tmp) / "export"
            zip_path = Path(tmp) / "export.zip"
            save_feedback(db, _make_payload("correct",   "p1"))
            save_feedback(db, _make_payload("incorrect", "p2"))
            export_for_training(db, out)
            zip_training_export(out, zip_path)

            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            self.assertIn("labels.csv", names)
            self.assertIn("retrain_log.json", names)
            png_files = [n for n in names if n.endswith(".png")]
            self.assertEqual(len(png_files), 2)


if __name__ == "__main__":
    unittest.main()
