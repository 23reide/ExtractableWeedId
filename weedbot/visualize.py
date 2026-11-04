"""Draw planner proposals and segmentation masks on an image.

The ``draw_scene`` function renders:
  - semi-transparent mask polygons (when provided by the model backend)
  - bounding boxes with class labels
  - annotated base / stem points
  - the selected proposal's dashed approach path

Mask rendering is optional: when ``masks`` is empty the function falls back to
the original box-only view so the ground-truth backend produces identical output
to the previous implementation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from .dataset import Scene

if TYPE_CHECKING:
    from .perception import MaskData

# Alpha for mask fills (0 = transparent, 255 = opaque)
_MASK_ALPHA = 90

# Plant-type colours (R, G, B)
_COLOURS: dict[str, tuple[int, int, int]] = {
    "crop": (30, 145, 80),
    "weed": (220, 55, 55),
    "soil": (130, 110, 80),
}


def _fill_colour(plant_type: str, alpha: int = _MASK_ALPHA) -> tuple[int, int, int, int]:
    r, g, b = _COLOURS.get(plant_type, (160, 160, 160))
    return (r, g, b, alpha)


def draw_scene(
    scene: Scene,
    selected_proposal: dict[str, Any] | None = None,
    masks: "list[MaskData] | None" = None,
) -> Image.Image:
    """Render *scene* onto its source image and return the annotated PIL image.

    Parameters
    ----------
    scene:
        The current scene (plants, image dimensions, paths).
    selected_proposal:
        If given, the dashed approach path and entry point are drawn.
    masks:
        Optional list of ``MaskData`` from the model backend.  When provided,
        semi-transparent coloured polygons are drawn for each detected instance.
        Pass ``None`` or an empty list to use bounding-box-only rendering.
    """
    image = Image.open(scene.image_path).convert("RGBA")
    font = ImageFont.load_default()

    # --- 1. Mask overlay layer (model backend only) -------------------------
    if masks:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        # Build a quick lookup: instance_id → plant_type
        plant_type_map = {p.instance_id: p.plant_type for p in scene.plants}
        for mask_data in masks:
            if len(mask_data.polygon) < 3:
                continue
            plant_type = plant_type_map.get(mask_data.instance_id, "soil")
            colour = _fill_colour(plant_type)
            flat_poly = [coord for pair in mask_data.polygon for coord in pair]
            overlay_draw.polygon(flat_poly, fill=colour)
        image = Image.alpha_composite(image, overlay)

    # Convert to RGB for drawing boxes / text (ImageDraw handles RGBA, but
    # the final image returned to Streamlit works better as RGB).
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    # --- 2. Bounding boxes + labels + stem dots ----------------------------
    for plant in scene.plants:
        left, top, right, bottom = plant.bbox
        outline = _COLOURS.get(plant.plant_type, (160, 160, 160))
        draw.rectangle((left, top, right, bottom), outline=outline, width=4)
        short = plant.plant_type[0].upper()
        label = f"{short}:{plant.instance_id.rsplit(':', 1)[-1]}"
        draw.text(
            (left + 2, max(0, top - 11)),
            label,
            fill=outline,
            font=font,
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
        bx, by = plant.base_xy
        r = 3
        draw.ellipse(
            (bx - r, by - r, bx + r, by + r),
            fill=(255, 215, 0),
            outline=(0, 0, 0),
        )

    # --- 3. Selected proposal: dashed approach path -------------------------
    if selected_proposal:
        tx, ty = selected_proposal["target_xy"]
        ex, ey = selected_proposal["entry_xy"]
        length = math.hypot(tx - ex, ty - ey)
        n = max(2, int(length / 5))
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1.0)
            p0 = (ex + (tx - ex) * t0, ey + (ty - ey) * t0)
            p1 = (ex + (tx - ex) * t1, ey + (ty - ey) * t1)
            draw.line((p0, p1), fill=(50, 110, 255), width=3)
        draw.ellipse(
            (tx - 6, ty - 6, tx + 6, ty + 6),
            outline=(50, 110, 255),
            width=3,
        )
        draw.text(
            (ex + 4, ey + 4),
            f"Approach {selected_proposal['approach_angle_deg']} deg",
            fill=(20, 70, 220),
            font=font,
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )

    return image
