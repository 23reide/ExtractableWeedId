"""Create a tiny synthetic CropAndWeed-shaped dataset to smoke-test the platform."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent / "demo_dataset"
IMAGE_DIR = ROOT / "images"
BBOX_DIR = ROOT / "bboxes" / "CropsOrWeed9"
MASK_DIR = ROOT / "labelIds" / "CropsOrWeed9"


def make_one(name: str, shift: int = 0) -> None:
    width, height = 640, 420
    image = Image.new("RGB", (width, height), (155, 130, 85))
    draw = ImageDraw.Draw(image)
    # Soil texture, deterministic and deliberately synthetic.
    for y in range(0, height, 12):
        draw.line((0, y, width, y), fill=(160 + (y // 12) % 8, 135, 91), width=1)
    objects = [
        # class_id 0 and 1 are crop classes in CropsOrWeed9; class_id 8 is weed.
        ("crop", 0, (115 + shift, 90, 205 + shift, 180), (160 + shift, 157)),
        ("crop", 1, (410, 220, 515, 315), (460, 292)),
        ("weed", 8, (275 + shift, 125, 338 + shift, 190), (307 + shift, 184)),
        ("weed", 8, (330, 290, 390, 355), (359, 348)),
    ]
    labels = np.zeros((height, width), dtype=np.uint8)
    for typ, label_id, (l, t, r, b), (sx, sy) in objects:
        fill = (45, 130, 45) if typ == "crop" else (75, 170, 70)
        # Elongated leaf-like ellipses and a stem, to resemble a basic overhead plant view.
        cx, cy = (l + r) // 2, (t + b) // 2
        draw.ellipse((l, cy - 14, cx + 10, cy + 16), fill=fill, outline=(30, 90, 35))
        draw.ellipse((cx - 4, t, r, cy + 3), fill=fill, outline=(30, 90, 35))
        draw.ellipse((cx - 18, cy - 7, cx + 16, b), fill=fill, outline=(30, 90, 35))
        draw.line((cx, cy, sx, sy), fill=(45, 100, 35), width=3)
        # A simple rectangular semantic-label region is enough for a demo fixture only.
        labels[max(0, t):min(height, b), max(0, l):min(width, r)] = label_id
    image.save(IMAGE_DIR / f"{name}.png")
    Image.fromarray(labels, mode="L").save(MASK_DIR / f"{name}.png")

    with (BBOX_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Left", "Top", "Right", "Bottom", "Label ID", "Stem X", "Stem Y"])
        for typ, label_id, bbox, stem in objects:
            writer.writerow([*bbox, label_id, *stem])


def main() -> None:
    for path in [IMAGE_DIR, BBOX_DIR, MASK_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    make_one("demo_0001", 0)
    make_one("demo_0002", 25)
    print(f"Created synthetic demo dataset at {ROOT}")


if __name__ == "__main__":
    main()
