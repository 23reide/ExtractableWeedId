"""Adapter for CropAndWeed-style data directories.

The adapter reads the dataset's per-image instance CSV files. For the
CropsOrWeed9 variant, label IDs 0..7 are crops and 8 is weed. This uses the
published plant boxes and stem annotations as ground truth so the geometry
planner can be tested independently from perception errors.
"""

from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PlantInstance:
    instance_id: str
    label_id: int
    plant_type: str  # "crop" or "weed"
    bbox: tuple[float, float, float, float]  # left, top, right, bottom
    base_xy: tuple[float, float]  # annotated stem location (or fallback)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["base_xy"] = list(self.base_xy)
        return data


@dataclass(frozen=True)
class Scene:
    image_id: str
    image_path: Path
    annotation_path: Path
    image_width: int
    image_height: int
    plants: tuple[PlantInstance, ...]
    dataset_variant: str

    @property
    def weeds(self) -> list[PlantInstance]:
        return [p for p in self.plants if p.plant_type == "weed"]

    @property
    def crops(self) -> list[PlantInstance]:
        return [p for p in self.plants if p.plant_type == "crop"]


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_image_paths(data_root: str | Path) -> list[Path]:
    """Find dataset images, preferring a dedicated ``images`` directory."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    preferred = root / "images"
    search_root = preferred if preferred.is_dir() else root
    paths = []
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = {part.lower() for part in path.parts}
        if "labelids" in parts or "masks" in parts or "segmentations" in parts:
            continue
        if "bboxes" in parts or "annotations" in parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: (p.parent.as_posix(), p.name.lower()))


@lru_cache(maxsize=8)
def _annotation_index(root_string: str) -> dict[str, tuple[str, ...]]:
    """Index annotation CSVs once; scanning the tree for every image is too slow on large datasets."""
    bbox_root = Path(root_string) / "bboxes"
    index: dict[str, list[str]] = {}
    if bbox_root.exists():
        for path in bbox_root.rglob("*.csv"):
            index.setdefault(path.stem, []).append(str(path.resolve()))
    return {stem: tuple(paths) for stem, paths in index.items()}


def find_bbox_csv(data_root: str | Path, image_path: str | Path, dataset_variant: str) -> Path | None:
    """Find the instance CSV matching an image, preferring the chosen variant."""
    root = Path(data_root).expanduser().resolve()
    stem = Path(image_path).stem
    index = _annotation_index(str(root))
    candidates = list(index.get(stem, ()))
    if not candidates:
        # A few dataset mirrors add suffixes to annotations; allow a conservative fallback.
        candidates = [csv_path for name, paths in index.items() if name.startswith(stem + "_") for csv_path in paths]
    if not candidates:
        return None

    variant_lower = dataset_variant.lower()
    candidates.sort(key=lambda p: (variant_lower not in str(Path(p).parent).lower(), len(Path(p).parts), p))
    return Path(candidates[0])


def _parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _row_to_plant(
    row: dict[str, Any],
    index: int,
    image_id: str,
    crop_label_ids: set[int],
    weed_label_ids: set[int],
    image_width: int,
    image_height: int,
) -> PlantInstance | None:
    # CSV headers from the official release: Left, Top, Right, Bottom, Label ID, Stem X, Stem Y
    norm_row = {_normalise_header(str(k)): v for k, v in row.items() if k is not None}

    def get(*aliases: str) -> Any:
        for alias in aliases:
            if _normalise_header(alias) in norm_row:
                return norm_row[_normalise_header(alias)]
        return None

    left = _parse_float(get("Left", "xmin", "x1"))
    top = _parse_float(get("Top", "ymin", "y1"))
    right = _parse_float(get("Right", "xmax", "x2"))
    bottom = _parse_float(get("Bottom", "ymax", "y2"))
    label_id = _parse_int(get("Label ID", "LabelId", "class_id", "label"))
    if None in (left, top, right, bottom) or label_id < 0:
        return None

    if label_id in crop_label_ids:
        plant_type = "crop"
    elif label_id in weed_label_ids:
        plant_type = "weed"
    else:
        # Ignore fallback/unknown classes instead of silently calling them weeds.
        return None

    left = min(max(float(left), 0.0), float(image_width - 1))
    right = min(max(float(right), 0.0), float(image_width - 1))
    top = min(max(float(top), 0.0), float(image_height - 1))
    bottom = min(max(float(bottom), 0.0), float(image_height - 1))
    if right <= left or bottom <= top:
        return None

    stem_x = _parse_float(get("Stem X", "StemX", "stem_x", "base_x"))
    stem_y = _parse_float(get("Stem Y", "StemY", "stem_y", "base_y"))
    # Dataset stem coordinates may be absent for some entries. The fallback is the
    # centre of the bottom edge of the bounding box and should be treated as uncertain.
    if stem_x is None or stem_y is None or (stem_x == 0 and stem_y == 0):
        stem_x = (left + right) / 2.0
        stem_y = bottom
    stem_x = min(max(stem_x, 0.0), float(image_width - 1))
    stem_y = min(max(stem_y, 0.0), float(image_height - 1))

    return PlantInstance(
        instance_id=f"{image_id}:{plant_type}:{index:03d}",
        label_id=label_id,
        plant_type=plant_type,
        bbox=(left, top, right, bottom),
        base_xy=(stem_x, stem_y),
    )


def load_scene(
    data_root: str | Path,
    image_path: str | Path,
    dataset_variant: str = "CropsOrWeed9",
    crop_label_ids: set[int] | None = None,
    weed_label_ids: set[int] | None = None,
) -> Scene:
    """Load one image and its instance annotations as a geometry scene.

    For the default CropsOrWeed9 variant, IDs 0-7 are crops and ID 8 is weeds.
    For the raw CropAndWeed variant, IDs 1-28 and 92-94 are treated as crops;
    29-91 and 95-99 are treated as weeds. Unknown IDs are ignored.
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    annotation_path = find_bbox_csv(data_root, image_path, dataset_variant)
    if annotation_path is None:
        raise FileNotFoundError(
            f"No per-image bbox CSV found for {image_path.name} under {Path(data_root) / 'bboxes'}. "
            "Check that the dataset annotations are installed and the variant name is correct."
        )

    if crop_label_ids is None or weed_label_ids is None:
        if dataset_variant.lower() == "cropandweed":
            default_crops = set(range(1, 29)) | {92, 93, 94}
            default_weeds = set(range(29, 92)) | {95, 96, 97, 98, 99}
        else:
            # Official CropsOrWeed9 mapping: 8 crop classes are grouped separately from weeds.
            default_crops = set(range(0, 8))
            default_weeds = {8}
        crop_label_ids = default_crops if crop_label_ids is None else crop_label_ids
        weed_label_ids = default_weeds if weed_label_ids is None else weed_label_ids

    with Image.open(image_path) as im:
        width, height = im.size

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(2048)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        plants: list[PlantInstance] = []
        for index, row in enumerate(reader):
            plant = _row_to_plant(
                row, index, image_path.stem, crop_label_ids, weed_label_ids, width, height
            )
            if plant is not None:
                plants.append(plant)

    # Handle CSVs that are present but do not contain a header.
    if not plants:
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            for index, row in enumerate(reader):
                if len(row) < 7:
                    continue
                plant = _row_to_plant(
                    dict(zip(["Left", "Top", "Right", "Bottom", "Label ID", "Stem X", "Stem Y"], row[:7])),
                    index, image_path.stem, crop_label_ids, weed_label_ids, width, height
                )
                if plant is not None:
                    plants.append(plant)

    return Scene(
        image_id=image_path.stem,
        image_path=image_path,
        annotation_path=annotation_path,
        image_width=width,
        image_height=height,
        plants=tuple(plants),
        dataset_variant=dataset_variant,
    )


def load_scenes(
    data_root: str | Path,
    dataset_variant: str = "CropsOrWeed9",
    max_images: int | None = None,
) -> list[Scene]:
    """Load available image/annotation pairs; skip images without annotations."""
    paths = find_image_paths(data_root)
    scenes: list[Scene] = []
    for path in paths:
        if find_bbox_csv(data_root, path, dataset_variant) is None:
            continue
        try:
            scene = load_scene(data_root, path, dataset_variant)
        except (OSError, ValueError, csv.Error):
            continue
        if scene.plants:
            scenes.append(scene)
        if max_images is not None and len(scenes) >= max_images:
            break
    return scenes
