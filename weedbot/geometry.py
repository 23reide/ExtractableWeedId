"""Interpretable 2D geometric extraction baseline.

This baseline treats crop bounding boxes as conservative obstacles. It samples
candidate approach paths toward each annotated weed base and estimates their
minimum clearance from those obstacles. Values are image pixels, not physical
millimetres. This is a planning heuristic, not a validated damage predictor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import PlantInstance, Scene

PLANNER_NAME = "bbox_clearance_v1"


@dataclass(frozen=True)
class PlannerConfig:
    gripper_radius_px: float = 8.0
    safety_margin_px: float = 3.0
    approach_length_px: float = 55.0
    angle_step_deg: int = 15
    path_sample_step_px: float = 2.0

    def __post_init__(self) -> None:
        if self.gripper_radius_px < 0:
            raise ValueError("gripper_radius_px must be non-negative")
        if self.safety_margin_px < 0:
            raise ValueError("safety_margin_px must be non-negative")
        if self.approach_length_px <= 0:
            raise ValueError("approach_length_px must be positive")
        if self.angle_step_deg <= 0 or 360 % self.angle_step_deg != 0:
            raise ValueError("angle_step_deg must be a positive divisor of 360")
        if self.path_sample_step_px <= 0:
            raise ValueError("path_sample_step_px must be positive")


@dataclass(frozen=True)
class CandidateApproach:
    angle_deg: int
    entry_xy: tuple[float, float]
    min_crop_clearance_px: float
    geometric_safe: bool
    entry_in_image: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["entry_xy"] = list(self.entry_xy)
        return result


def point_to_bbox_distance(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    """Euclidean distance from a point to an axis-aligned bbox (zero inside)."""
    left, top, right, bottom = bbox
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return math.hypot(dx, dy)


def _path_clearance(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    crop_bboxes: list[tuple[float, float, float, float]],
    sample_step_px: float,
) -> float:
    if not crop_bboxes:
        return float("inf")
    sx, sy = start_xy
    tx, ty = target_xy
    length = math.hypot(tx - sx, ty - sy)
    count = max(2, int(math.ceil(length / sample_step_px)) + 1)
    min_clearance = float("inf")
    for idx in range(count):
        t = idx / (count - 1)
        x = sx + (tx - sx) * t
        y = sy + (ty - sy) * t
        for bbox in crop_bboxes:
            min_clearance = min(min_clearance, point_to_bbox_distance(x, y, bbox))
            if min_clearance == 0:
                return 0.0
    return min_clearance


def evaluate_approaches(
    weed: PlantInstance,
    crops: list[PlantInstance],
    config: PlannerConfig,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[CandidateApproach]:
    """Evaluate equally spaced approach angles, from all directions around a weed.

    Angle 0° points right in image coordinates; positive angles increase
    clockwise because image y coordinates increase downward. The proposed path
    runs from ``entry_xy`` toward the weed base.
    """
    crop_bboxes = [crop.bbox for crop in crops]
    base_x, base_y = weed.base_xy
    safe_threshold = config.gripper_radius_px + config.safety_margin_px
    results = []

    for angle in range(0, 360, config.angle_step_deg):
        radians = math.radians(angle)
        dx, dy = math.cos(radians), math.sin(radians)
        entry = (
            base_x + dx * config.approach_length_px,
            base_y + dy * config.approach_length_px,
        )
        # If image dimensions are known, reject approach start points outside the image.
        inside = (image_width is None or 0 <= entry[0] < image_width) and (
            image_height is None or 0 <= entry[1] < image_height
        )
        clearance = _path_clearance(entry, weed.base_xy, crop_bboxes, config.path_sample_step_px)
        results.append(
            CandidateApproach(
                angle_deg=angle,
                entry_xy=entry,
                min_crop_clearance_px=clearance,
                geometric_safe=bool(inside and clearance >= safe_threshold),
                entry_in_image=bool(inside),
            )
        )
    return results


def propose_for_weed(scene: Scene, weed: PlantInstance, config: PlannerConfig) -> dict[str, Any]:
    candidates = evaluate_approaches(
        weed, scene.crops, config, image_width=scene.image_width, image_height=scene.image_height
    )
    # Prefer safe paths; within each group, maximise clearance and then use a stable angle tie-break.
    best = max(
        candidates,
        key=lambda c: (c.geometric_safe, c.entry_in_image, c.min_crop_clearance_px, -c.angle_deg),
    )
    safe_candidates = [candidate for candidate in candidates if candidate.geometric_safe]
    distances = [point_to_bbox_distance(*weed.base_xy, crop.bbox) for crop in scene.crops]
    nearest_index = min(range(len(distances)), key=distances.__getitem__) if distances else None
    nearest_crop = scene.crops[nearest_index] if nearest_index is not None else None

    return {
        "proposal_id": f"{scene.image_id}::{weed.instance_id}::{PLANNER_NAME}",
        "image_id": scene.image_id,
        "image_path": str(scene.image_path),
        "annotation_path": str(scene.annotation_path),
        "dataset_variant": scene.dataset_variant,
        "target_weed_id": weed.instance_id,
        "target_label_id": weed.label_id,
        "target_bbox_xyxy": list(weed.bbox),
        "target_xy": [round(weed.base_xy[0], 3), round(weed.base_xy[1], 3)],
        "nearest_crop_id": nearest_crop.instance_id if nearest_crop else None,
        "nearest_crop_bbox_distance_px": round(distances[nearest_index], 3) if nearest_index is not None else None,
        "planner": PLANNER_NAME,
        "planner_config": asdict(config),
        "decision": "attempt" if best.geometric_safe else "skip",
        "geometric_safe": best.geometric_safe,
        "approach_angle_deg": best.angle_deg,
        "entry_xy": [round(best.entry_xy[0], 3), round(best.entry_xy[1], 3)],
        "min_crop_clearance_px": None if math.isinf(best.min_crop_clearance_px) else round(best.min_crop_clearance_px, 3),
        "safe_candidate_count": len(safe_candidates),
        "candidate_approaches": [candidate.to_dict() for candidate in candidates],
        "limitations": [
            "2D image coordinates only; pixel distances are not calibrated physical distances.",
            "Crop bounding boxes are conservative approximations of plant geometry.",
            "Geometric safety does not prove physical extraction success or crop damage.",
        ],
    }


def propose_for_scene(scene: Scene, config: PlannerConfig | None = None) -> list[dict[str, Any]]:
    config = config or PlannerConfig()
    return [propose_for_weed(scene, weed, config) for weed in scene.weeds]
