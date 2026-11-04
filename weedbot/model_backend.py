"""YOLOv8-nano-seg model backend.

Runs a YOLOv8 segmentation model on a raw image and converts detections into
the ``Scene`` / ``PlantInstance`` format expected by the geometry planner.

Weights are loaded from ``config.yaml → model_weights_path``.  If no custom
weights are configured, the pretrained COCO yolov8n-seg weights shipped with
ultralytics are used as a placeholder until a fine-tuned checkpoint is available.

Class-name → plant-type mapping is controlled by the ``label_map`` key in
``config.yaml`` (lowercased class name → "crop" / "weed" / "soil").  Any
detected class that is not in the map is silently ignored, so spurious
detections from COCO weights do not corrupt the scene.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import PlantInstance, Scene
from .perception import MaskData, PerceptionBackend, PerceptionResult

# Default weights used when model_weights_path is empty / unset.
_DEFAULT_WEIGHTS = "yolov8n-seg.pt"

# Default label map — covers the fine-tuned class names the Colab notebook
# produces as well as loose COCO approximations for pre-training.
_DEFAULT_LABEL_MAP: dict[str, str] = {
    "crop": "crop",
    "crops": "crop",
    "potted plant": "crop",
    "plant": "crop",
    "weed": "weed",
    "weeds": "weed",
    "grass": "weed",
    "soil": "soil",
    "dirt": "soil",
    "ground": "soil",
}


def _polygon_from_mask(mask_array: "np.ndarray") -> list[tuple[float, float]]:
    """Extract a simplified polygon from a boolean mask via contour tracing.

    Uses OpenCV if available for accuracy; falls back to a tight bounding-box
    rectangle so the visualiser always has *something* to render.
    """
    try:
        import cv2  # type: ignore[import]

        contours, _ = cv2.findContours(
            mask_array.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return []
        # Use the largest contour and approximate it to reduce point count.
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, closed=True)
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    except ImportError:
        pass

    # Fallback: bounding rectangle of the mask
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if not rows.any():
        return []
    rmin, rmax = int(np.where(rows)[0][[0, -1]])
    cmin, cmax = int(np.where(cols)[0][[0, -1]])
    return [
        (float(cmin), float(rmin)),
        (float(cmax), float(rmin)),
        (float(cmax), float(rmax)),
        (float(cmin), float(rmax)),
    ]


def _stem_from_mask(mask_array: "np.ndarray") -> tuple[float, float]:
    """Estimate the stem / base position from a binary mask.

    Heuristic: centroid x-coordinate at the bottom-most 20 % of the mask rows.
    This approximates where a stem would emerge from the soil.
    """
    ys, xs = np.where(mask_array)
    if len(ys) == 0:
        return (0.0, 0.0)
    y_max = int(ys.max())
    y_thresh = y_max - max(1, int((y_max - ys.min()) * 0.2))
    bottom_xs = xs[ys >= y_thresh]
    if len(bottom_xs) == 0:
        return (float(xs.mean()), float(y_max))
    return (float(bottom_xs.mean()), float(y_max))


class ModelBackend(PerceptionBackend):
    """YOLOv8-nano-seg inference backend."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        label_map: dict[str, str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Parameters
        ----------
        weights_path:
            Path to a ``.pt`` weights file.  ``None`` or empty string → default
            pretrained COCO weights (``yolov8n-seg.pt``).
        label_map:
            Maps lowercased YOLO class names to ``"crop"``, ``"weed"``, or
            ``"soil"``.  Merged with built-in defaults; caller entries win.
        conf_threshold:
            Minimum detection confidence to keep.
        iou_threshold:
            Non-maximum-suppression IoU threshold.
        """
        from ultralytics import YOLO  # type: ignore[import]

        if weights_path:
            resolved = Path(weights_path).expanduser()
            if not resolved.is_absolute():
                # Resolve relative paths from the project root (parent of weedbot/).
                project_root = Path(__file__).resolve().parent.parent
                resolved = (project_root / resolved).resolve()
            self._model = YOLO(str(resolved))
        else:
            self._model = YOLO(_DEFAULT_WEIGHTS)

        # Merge caller label map over built-in defaults.
        merged: dict[str, str] = dict(_DEFAULT_LABEL_MAP)
        if label_map:
            merged.update({k.lower(): v for k, v in label_map.items()})
        self._label_map = merged

        self._conf = conf_threshold
        self._iou = iou_threshold

    # ------------------------------------------------------------------
    # PerceptionBackend implementation
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> PerceptionResult:
        """Run YOLOv8-seg on *image_path* and return a PerceptionResult."""
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        # Run inference (device="cpu" ensures laptop compatibility).
        results = self._model(
            str(image_path),
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )

        plants: list[PlantInstance] = []
        masks: list[MaskData] = []
        image_id = image_path.stem

        result = results[0]
        names: dict[int, str] = result.names  # {class_id: name}

        has_masks = (
            result.masks is not None
            and result.masks.data is not None
            and len(result.masks.data) > 0
        )

        boxes = result.boxes
        n_detections = len(boxes) if boxes is not None else 0

        for idx in range(n_detections):
            class_id = int(boxes.cls[idx].item())
            class_name = names.get(class_id, "").lower()
            plant_type = self._label_map.get(class_name)
            if plant_type not in ("crop", "weed", "soil"):
                continue  # ignore unknown classes

            # Bounding box (xyxy format, pixel coords)
            xyxy = boxes.xyxy[idx].cpu().numpy()
            left, top, right, bottom = (
                float(max(0.0, xyxy[0])),
                float(max(0.0, xyxy[1])),
                float(min(width - 1, xyxy[2])),
                float(min(height - 1, xyxy[3])),
            )
            if right <= left or bottom <= top:
                continue

            instance_id = f"{image_id}:{plant_type}:{idx:03d}"

            # Mask polygon + stem estimate
            polygon: list[tuple[float, float]] = []
            if has_masks and idx < len(result.masks.data):
                raw_mask = result.masks.data[idx].cpu().numpy()
                # Masks may be at a different resolution; resize to image dims.
                if raw_mask.shape != (height, width):
                    from PIL import Image as _PILImage  # noqa: PLC0415
                    mask_img = _PILImage.fromarray((raw_mask * 255).astype(np.uint8), mode="L")
                    mask_img = mask_img.resize((width, height), _PILImage.NEAREST)
                    raw_mask = np.array(mask_img) > 127
                bool_mask = raw_mask > 0.5
                polygon = _polygon_from_mask(bool_mask)
                stem_xy = _stem_from_mask(bool_mask)
            else:
                # No mask available: derive stem from bbox bottom-centre.
                stem_xy = ((left + right) / 2.0, bottom)

            plants.append(
                PlantInstance(
                    instance_id=instance_id,
                    label_id=class_id,
                    plant_type=plant_type,
                    bbox=(left, top, right, bottom),
                    base_xy=stem_xy,
                )
            )
            masks.append(MaskData(instance_id=instance_id, polygon=polygon))

        scene = Scene(
            image_id=image_id,
            image_path=image_path,
            annotation_path=image_path,  # no separate annotation file for model output
            image_width=width,
            image_height=height,
            plants=tuple(plants),
            dataset_variant="model",
        )
        return PerceptionResult(scene=scene, masks=masks)
