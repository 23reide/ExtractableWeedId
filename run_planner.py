"""Batch-export geometric action proposals to JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weedbot.dataset import load_scenes
from weedbot.geometry import PlannerConfig, propose_for_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="CropAndWeed prepared data directory")
    parser.add_argument("--variant", default="CropsOrWeed9", help="Annotation variant; default: CropsOrWeed9")
    parser.add_argument("--out", default="outputs/proposals.jsonl", help="Destination JSONL file")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for a quick test")
    parser.add_argument("--gripper-radius-px", type=float, default=8.0)
    parser.add_argument("--safety-margin-px", type=float, default=3.0)
    parser.add_argument("--approach-length-px", type=float, default=55.0)
    parser.add_argument("--angle-step-deg", type=int, default=15)
    args = parser.parse_args()

    config = PlannerConfig(
        gripper_radius_px=args.gripper_radius_px,
        safety_margin_px=args.safety_margin_px,
        approach_length_px=args.approach_length_px,
        angle_step_deg=args.angle_step_deg,
    )
    scenes = load_scenes(args.data_root, args.variant, args.max_images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_count = 0
    with out_path.open("w", encoding="utf-8") as stream:
        for scene in scenes:
            for proposal in propose_for_scene(scene, config):
                stream.write(json.dumps(proposal) + "\n")
                proposal_count += 1
    print(f"Loaded {len(scenes)} image(s); exported {proposal_count} proposal(s) to {out_path.resolve()}")
    if not scenes:
        print("No usable image/annotation pairs found. Check --data-root and --variant.")


if __name__ == "__main__":
    main()
