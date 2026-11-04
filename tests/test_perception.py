"""Tests for the perception abstraction layer."""
import unittest
from pathlib import Path

from weedbot.dataset import load_scene
from weedbot.perception import GroundTruthBackend, PerceptionResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo_dataset"
DEMO_IMAGE = DEMO_ROOT / "images" / "demo_0001.png"


@unittest.skipUnless(DEMO_IMAGE.exists(), "demo dataset not generated — run make_demo_data.py")
class GroundTruthBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = GroundTruthBackend(
            data_root=DEMO_ROOT, dataset_variant="CropsOrWeed9"
        )

    def test_returns_perception_result(self):
        result = self.backend.predict(DEMO_IMAGE)
        self.assertIsInstance(result, PerceptionResult)

    def test_scene_matches_load_scene(self):
        """GroundTruthBackend.predict() must return the same scene as load_scene()."""
        result = self.backend.predict(DEMO_IMAGE)
        direct = load_scene(DEMO_ROOT, DEMO_IMAGE, "CropsOrWeed9")
        self.assertEqual(result.scene.image_id, direct.image_id)
        self.assertEqual(len(result.scene.plants), len(direct.plants))
        self.assertEqual(result.scene.image_width, direct.image_width)
        self.assertEqual(result.scene.image_height, direct.image_height)

    def test_no_masks_for_groundtruth(self):
        result = self.backend.predict(DEMO_IMAGE)
        self.assertEqual(result.masks, [])

    def test_scene_has_weeds_and_crops(self):
        result = self.backend.predict(DEMO_IMAGE)
        self.assertTrue(len(result.scene.weeds) > 0, "Expected at least one weed")
        self.assertTrue(len(result.scene.crops) > 0, "Expected at least one crop")


if __name__ == "__main__":
    unittest.main()
