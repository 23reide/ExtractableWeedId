import tempfile
import unittest
from pathlib import Path

from weedbot.dataset import load_scenes
from weedbot.storage import export_feedback_csv, save_feedback

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DatasetAndStorageTests(unittest.TestCase):
    def test_load_synthetic_dataset(self):
        scenes = load_scenes(PROJECT_ROOT / "demo_dataset", "CropsOrWeed9")
        self.assertEqual(len(scenes), 2)
        self.assertEqual(len(scenes[0].weeds), 2)
        self.assertEqual(len(scenes[0].crops), 2)
        self.assertTrue(all(0 <= weed.base_xy[0] < scenes[0].image_width for weed in scenes[0].weeds))

    def test_feedback_persists_and_exports(self):
        payload = {
            "image_id": "unit_test_image",
            "image_path": "/some/image.png",
            "proposal_id": "unit_test_image::weed_1::bbox_clearance_v1",
            "planner_name": "bbox_clearance_v1",
            "planner_config": {"gripper_radius_px": 8},
            "proposal": {"decision": "attempt", "target_xy": [3, 4]},
            "valid_weed_target": "yes",
            "approach_safety": "safe",
            "weed_removed": "cannot_tell",
            "crop_damaged": "uncertain",
            "damage_severity": "unknown",
            "annotator_confidence": "medium",
            "notes": "unit test",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.sqlite"
            csv_path = Path(temp_dir) / "feedback.csv"
            feedback_id = save_feedback(db_path, payload)
            self.assertEqual(feedback_id, 1)
            rows = export_feedback_csv(db_path, csv_path)
            self.assertEqual(rows, 1)
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("unit_test_image", content)
            self.assertIn("unit test", content)


if __name__ == "__main__":
    unittest.main()
