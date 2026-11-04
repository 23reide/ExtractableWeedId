"""Export labeled feedback as image patches for fine-tuning.

``export_for_training`` reads all feedback rows where ``pull_decision_correct``
is ``"correct"`` or ``"incorrect"``, crops a fixed region around each target
weed from its source image, writes the patch as a PNG, and produces a
``labels.csv`` that the Colab notebook can consume directly.

Output layout::

    <output_dir>/
        images/
            <proposal_id_safe>.png
            ...
        labels.csv          # one row per patch
        retrain_log.json    # timestamp + count (also written to outputs/)
"""
from __future__ import annotations

import csv
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .storage import connect as _db_connect

# Patch size (pixels).  The weed target is centred in this square crop.
PATCH_SIZE = 256

_CSV_FIELDNAMES = [
    "patch_path",
    "image_id",
    "proposal_id",
    "model_decision",
    "pull_decision_correct",
    "annotated_at",
]


def _safe_filename(proposal_id: str) -> str:
    """Convert a proposal_id into a safe filename stem (replace :: and : with _)."""
    return proposal_id.replace("::", "_").replace(":", "_")


def export_for_training(
    db_path: str | Path,
    output_dir: str | Path,
    patch_size: int = PATCH_SIZE,
) -> int:
    """Export labeled feedback rows as image patches.

    Parameters
    ----------
    db_path:
        Path to the SQLite feedback database.
    output_dir:
        Destination directory.  Created if it does not exist.  Existing patches
        are overwritten so re-running is idempotent.
    patch_size:
        Side length (pixels) of the square crop centred on the target weed.

    Returns
    -------
    int
        Number of patches successfully written.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    conn = _db_connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM feedback "
            "WHERE pull_decision_correct IN ('correct','incorrect') "
            "ORDER BY feedback_id"
        ).fetchall()
    finally:
        conn.close()

    exported: list[dict[str, str]] = []
    half = patch_size // 2

    for row in rows:
        row_dict = dict(row)
        proposal_json = row_dict.get("proposal_json", "{}")
        try:
            proposal: dict[str, Any] = json.loads(proposal_json)
        except (json.JSONDecodeError, TypeError):
            continue

        image_path = Path(row_dict.get("image_path", ""))
        if not image_path.exists():
            continue

        target_xy = proposal.get("target_xy")
        if not target_xy or len(target_xy) < 2:
            continue

        cx, cy = float(target_xy[0]), float(target_xy[1])

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                # Clamp the crop window to image bounds.
                left = max(0, int(cx - half))
                top = max(0, int(cy - half))
                right = min(w, left + patch_size)
                bottom = min(h, top + patch_size)
                patch = img.crop((left, top, right, bottom))
                # Pad to exact patch_size if the crop hit an image edge.
                if patch.size != (patch_size, patch_size):
                    padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                    padded.paste(patch, (0, 0))
                    patch = padded
        except Exception:
            continue

        proposal_id = row_dict.get("proposal_id", f"row_{row_dict['feedback_id']}")
        filename = _safe_filename(proposal_id) + ".png"
        patch_path = images_dir / filename
        patch.save(patch_path)

        exported.append({
            "patch_path": patch_path.relative_to(output_dir).as_posix(),
            "image_id": row_dict.get("image_id", ""),
            "proposal_id": proposal_id,
            "model_decision": proposal.get("decision", ""),
            "pull_decision_correct": row_dict.get("pull_decision_correct", ""),
            "annotated_at": row_dict.get("created_at", ""),
        })

    # Write labels.csv
    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(exported)

    count = len(exported)

    # Write retrain_log.json inside the export directory.
    log_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": count,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "retrain_log.json").write_text(
        json.dumps(log_data, indent=2), encoding="utf-8"
    )

    return count


def zip_training_export(export_dir: str | Path, zip_path: str | Path) -> Path:
    """Zip the contents of *export_dir* to *zip_path* and return the zip path."""
    export_dir = Path(export_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in export_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(export_dir))
    return zip_path
