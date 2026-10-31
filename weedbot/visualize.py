"""Draw planner proposals on an image for the annotation interface."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .dataset import Scene


def draw_scene(scene: Scene, selected_proposal: dict[str, Any] | None = None) -> Image.Image:
    image = Image.open(scene.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Crop boxes are orange; weed boxes are red. The base point is explicitly marked.
    for plant in scene.plants:
        left, top, right, bottom = plant.bbox
        is_crop = plant.plant_type == "crop"
        outline = (30, 145, 80) if is_crop else (220, 55, 55)
        draw.rectangle((left, top, right, bottom), outline=outline, width=2)
        label = f"{'C' if is_crop else 'W'}:{plant.instance_id.rsplit(':', 1)[-1]}"
        draw.text((left + 2, max(0, top - 11)), label, fill=outline, font=font, stroke_width=1, stroke_fill=(255, 255, 255))
        bx, by = plant.base_xy
        r = 3
        draw.ellipse((bx-r, by-r, bx+r, by+r), fill=(255, 215, 0), outline=(0, 0, 0))

    if selected_proposal:
        tx, ty = selected_proposal["target_xy"]
        ex, ey = selected_proposal["entry_xy"]
        # Dashed path from approach entry to the proposed extraction point.
        length = math.hypot(tx - ex, ty - ey)
        n = max(2, int(length / 5))
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1.0)
            p0 = (ex + (tx-ex)*t0, ey + (ty-ey)*t0)
            p1 = (ex + (tx-ex)*t1, ey + (ty-ey)*t1)
            draw.line((p0, p1), fill=(50, 110, 255), width=3)
        draw.ellipse((tx-6, ty-6, tx+6, ty+6), outline=(50, 110, 255), width=3)
        draw.text((ex + 4, ey + 4), f"Approach {selected_proposal['approach_angle_deg']} deg", fill=(20, 70, 220), font=font, stroke_width=1, stroke_fill=(255,255,255))

    return image
