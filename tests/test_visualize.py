"""Tests for the visualise module including mask overlay rendering."""
import unittest
from pathlib import Path

from PIL import Image

from weedbot.perception import MaskData
from weedbot.visualize import draw_scene

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo_dataset"
DEMO_IMAGE = DEMO_ROOT / "images" / "demo_0001.png"


@unittest.skipUnless(DEMO_IMAGE.exists(), "demo dataset not generated — run make_demo_data.py")
class VisualizeTests(unittest.TestCase):
    def _load_scene(self):
        from weedbot.dataset import load_scene
        return load_scene(DEMO_ROOT, DEMO_IMAGE, "CropsOrWeed9")

    def test_draw_scene_no_masks(self):
        scene = self._load_scene()
        result = draw_scene(scene)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (scene.image_width, scene.image_height))

    def test_draw_scene_with_empty_masks(self):
        scene = self._load_scene()
        result = draw_scene(scene, masks=[])
        self.assertIsInstance(result, Image.Image)

    def test_draw_scene_with_mock_masks(self):
        scene = self._load_scene()
        # Build a triangular mock mask for the first plant.
        first_plant = scene.plants[0]
        l, t, r, b = first_plant.bbox
        mock_mask = MaskData(
            instance_id=first_plant.instance_id,
            polygon=[(l, t), (r, t), ((l + r) / 2, b)],
        )
        result = draw_scene(scene, masks=[mock_mask])
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (scene.image_width, scene.image_height))
        # The image should be in RGB mode after compositing.
        self.assertEqual(result.mode, "RGB")

    def test_draw_scene_with_proposal(self):
        from weedbot.geometry import PlannerConfig, propose_for_scene
        scene = self._load_scene()
        proposals = propose_for_scene(scene, PlannerConfig())
        if proposals:
            result = draw_scene(scene, selected_proposal=proposals[0])
            self.assertIsInstance(result, Image.Image)


if __name__ == "__main__":
    unittest.main()
