import unittest

from weedbot.dataset import PlantInstance
from weedbot.geometry import PlannerConfig, evaluate_approaches, point_to_bbox_distance


class GeometryTests(unittest.TestCase):
    def test_point_to_bbox_distance(self):
        bbox = (10.0, 10.0, 20.0, 20.0)
        self.assertEqual(point_to_bbox_distance(15, 15, bbox), 0.0)
        self.assertAlmostEqual(point_to_bbox_distance(25, 15, bbox), 5.0)
        self.assertAlmostEqual(point_to_bbox_distance(23, 24, bbox), 5.0)

    def test_approach_with_clear_open_side(self):
        weed = PlantInstance("w1", 8, "weed", (40, 40, 55, 55), (47, 50))
        crop = PlantInstance("c1", 0, "crop", (60, 40, 85, 65), (72, 60))
        results = evaluate_approaches(weed, [crop], PlannerConfig(
            gripper_radius_px=3, safety_margin_px=1, approach_length_px=20,
            angle_step_deg=90, path_sample_step_px=1,
        ))
        self.assertEqual(len(results), 4)
        self.assertTrue(any(result.geometric_safe for result in results))
        self.assertTrue(any(not result.geometric_safe for result in results))

    def test_zero_crop_obstacles_all_directions_safe(self):
        weed = PlantInstance("w1", 8, "weed", (40, 40, 55, 55), (47, 50))
        results = evaluate_approaches(weed, [], PlannerConfig(angle_step_deg=45))
        self.assertTrue(all(result.geometric_safe for result in results))


if __name__ == "__main__":
    unittest.main()
