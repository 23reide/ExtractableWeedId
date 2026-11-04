"""Perception backend abstraction.

All perception sources — ground-truth CSV annotations and live model inference —
implement the ``PerceptionBackend`` interface so the planner and UI remain fully
decoupled from the underlying source.

The ``PerceptionResult`` dataclass bundles the geometry-ready ``Scene`` with
optional per-instance mask polygons so the visualiser can render overlays when
they are available (model backend) and fall back to bounding-box rendering when
they are not (ground-truth backend).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import Scene, load_scene


# ---------------------------------------------------------------------------
# PerceptionResult
# ---------------------------------------------------------------------------

@dataclass
class MaskData:
    """Mask information for a single plant instance.

    ``instance_id`` matches ``PlantInstance.instance_id`` so the visualiser can
    look up the plant type for colouring.

    ``polygon`` is a flat list of (x, y) pairs in image pixel coordinates, e.g.
    ``[(x0, y0), (x1, y1), ...]``.  May be empty if only a bounding box is
    available.
    """
    instance_id: str
    polygon: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PerceptionResult:
    """Output from any PerceptionBackend.

    ``scene`` is always populated and is the primary input for the geometry
    planner.  ``masks`` is populated by the model backend and may be empty for
    the ground-truth backend.
    """
    scene: Scene
    masks: list[MaskData] = field(default_factory=list)

    def mask_for(self, instance_id: str) -> MaskData | None:
        """Return the mask matching *instance_id*, or None."""
        for m in self.masks:
            if m.instance_id == instance_id:
                return m
        return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PerceptionBackend(abc.ABC):
    """Minimal interface every perception source must implement."""

    @abc.abstractmethod
    def predict(self, image_path: str | Path) -> PerceptionResult:
        """Run perception on *image_path* and return a PerceptionResult.

        The returned ``PerceptionResult.scene`` must have its ``image_path``,
        ``image_width``, ``image_height``, and ``plants`` fields populated so
        the downstream geometry planner can run without modification.
        """


# ---------------------------------------------------------------------------
# Ground-truth backend (wraps existing load_scene)
# ---------------------------------------------------------------------------

class GroundTruthBackend(PerceptionBackend):
    """Uses pre-labelled CropAndWeed annotation CSVs as the perception source.

    This backend is identical in behaviour to the original ``load_scene`` call
    and exists so the planner and UI can use the same code path regardless of
    which backend is active.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset_variant: str = "CropsOrWeed9",
        crop_label_ids: set[int] | None = None,
        weed_label_ids: set[int] | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.dataset_variant = dataset_variant
        self.crop_label_ids = crop_label_ids
        self.weed_label_ids = weed_label_ids

    def predict(self, image_path: str | Path) -> PerceptionResult:
        scene = load_scene(
            self.data_root,
            image_path,
            dataset_variant=self.dataset_variant,
            crop_label_ids=self.crop_label_ids,
            weed_label_ids=self.weed_label_ids,
        )
        # Ground-truth backend has no mask polygons; visualiser falls back to boxes.
        return PerceptionResult(scene=scene, masks=[])


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML project config.  Returns an empty dict if the file is missing."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def backend_from_config(config: dict[str, Any] | None = None) -> PerceptionBackend:
    """Instantiate the backend specified in *config* (or loaded from ``config.yaml``).

    Falls back to ``GroundTruthBackend`` pointing at ``demo_dataset`` if no
    config is found, so the app always starts without error.
    """
    if config is None:
        config = load_config()

    backend_name = str(config.get("backend", "groundtruth")).strip().lower()

    if backend_name == "model":
        # Lazy import so the ground-truth path never requires ultralytics to be installed.
        from .model_backend import ModelBackend  # noqa: PLC0415

        weights = config.get("model_weights_path") or ""
        weights = weights.strip() if isinstance(weights, str) else ""
        label_map: dict[str, str] = config.get("label_map") or {}
        return ModelBackend(
            weights_path=weights if weights else None,
            label_map=label_map,
        )

    # Default: ground-truth
    data_root = config.get("data_root", "demo_dataset")
    variant = config.get("dataset_variant", "CropsOrWeed9")
    return GroundTruthBackend(data_root=data_root, dataset_variant=variant)
