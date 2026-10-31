"""Streamlit annotation UI for reviewing simulated weed-extraction proposals."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from weedbot.dataset import load_scene
from weedbot.storage import export_feedback_csv, save_feedback
from weedbot.visualize import draw_scene

st.set_page_config(page_title="Weed Extraction Planner Review", layout="wide")
st.title("Weed-extraction planning: human review")
st.caption(
    "This is a 2D software benchmark. Labels describe an annotator's judgment of a proposed action, "
    "not verified physical extraction outcomes. Coordinates and clearances are image pixels."
)

with st.sidebar:
    st.header("Proposal review")
    proposal_file_name = st.text_input("Proposal file in outputs", value="demo_proposals.jsonl")
    st.divider()
    db_path = st.text_input("Feedback database", value="outputs/feedback.sqlite")

proposal_file_name = proposal_file_name.strip()
proposal_file = Path("outputs") / proposal_file_name
if Path(proposal_file_name).name != proposal_file_name or proposal_file.suffix.lower() != ".jsonl":
    st.error("Enter the name of a `.jsonl` proposal file located in the `outputs` folder.")
    st.stop()

try:
    with proposal_file.open("r", encoding="utf-8") as stream:
        proposals = []
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                proposal = json.loads(line)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON in `{proposal_file}` on line {line_number}: {exc}")
                st.stop()
            if not isinstance(proposal, dict):
                st.error(f"Expected a proposal object in `{proposal_file}` on line {line_number}.")
                st.stop()
            proposals.append(proposal)
except OSError as exc:
    st.error(f"Could not open proposal file `{proposal_file}`: {exc}")
    st.stop()

if not proposals:
    st.info(f"No proposals found in `{proposal_file}`.")
    st.stop()

required_fields = {
    "proposal_id",
    "image_path",
    "annotation_path",
    "dataset_variant",
    "target_weed_id",
    "target_xy",
    "entry_xy",
    "decision",
    "approach_angle_deg",
    "safe_candidate_count",
    "candidate_approaches",
    "planner",
    "planner_config",
}
for index, item in enumerate(proposals, start=1):
    missing_fields = required_fields.difference(item)
    if missing_fields:
        st.error(
            f"Proposal {index} in `{proposal_file}` is missing required field(s): "
            f"{', '.join(sorted(missing_fields))}."
        )
        st.stop()

proposal_labels = {
    proposal["proposal_id"]: (
        f"{proposal['target_weed_id']} — {proposal['decision'].upper()} — "
        f"angle {proposal['approach_angle_deg']}° — "
        f"{proposal['safe_candidate_count']} safe direction(s)"
    )
    for proposal in proposals
}
selected_proposal_id = st.selectbox(
    "Proposed extraction action",
    options=[proposal["proposal_id"] for proposal in proposals],
    format_func=lambda value: proposal_labels[value],
)
proposal = next(p for p in proposals if p["proposal_id"] == selected_proposal_id)

image_path = Path(proposal["image_path"]).expanduser()
annotation_path = Path(proposal["annotation_path"]).expanduser()
bbox_root = next((parent for parent in annotation_path.parents if parent.name.lower() == "bboxes"), None)
if bbox_root is None:
    st.error(f"Could not determine the dataset root from annotation path `{annotation_path}`.")
    st.stop()

try:
    scene = load_scene(bbox_root.parent, image_path, proposal["dataset_variant"])
except (OSError, ValueError) as exc:
    st.error(f"Could not load the scene referenced by this proposal: {exc}")
    st.stop()

left, right = st.columns([1.6, 1])
with left:
    st.image(draw_scene(scene, proposal), caption="Green boxes: crop · Red boxes: weed · Yellow dot: estimated/annotated base · Blue line: proposed approach", use_container_width=True)
with right:
    st.subheader("Action proposal")
    st.json({key: value for key, value in proposal.items() if key != "candidate_approaches" and key != "limitations"})
    with st.expander("Candidate approach details"):
        st.dataframe(
            [
                {
                    "angle_deg": c["angle_deg"],
                    "entry_x": round(c["entry_xy"][0], 1),
                    "entry_y": round(c["entry_xy"][1], 1),
                    "clearance_px": c["min_crop_clearance_px"],
                    "geometric_safe": c["geometric_safe"],
                }
                for c in proposal["candidate_approaches"]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.subheader("Annotate this proposed action")
st.write("Judge what appears plausible from the image. Choose `uncertain` when a static view does not support a confident judgment.")
with st.form(key=f"feedback-{proposal['proposal_id']}"):
    valid_target = st.radio("Is the target actually a weed suitable for removal?", ["yes", "no", "uncertain"], horizontal=True)
    approach_safety = st.radio("Does the proposed path appear safe for neighboring crops?", ["safe", "unsafe", "uncertain"], horizontal=True)
    weed_removed = st.radio("Would the weed be removable using this proposed action?", ["likely_yes", "likely_no", "cannot_tell"], horizontal=True)
    crop_damaged = st.radio("Would a neighboring crop likely be damaged?", ["yes", "no", "uncertain"], horizontal=True)
    damage_severity = st.selectbox("Estimated crop-damage severity", ["none", "minor", "moderate", "severe", "unknown"])
    confidence = st.select_slider("Annotator confidence", options=["low", "medium", "high"], value="medium")
    notes = st.text_area("Notes (optional)", placeholder="What made this action appear safe, risky, or uncertain?")
    saved = st.form_submit_button("Save feedback", type="primary")
    if saved:
        payload = {
            "image_id": scene.image_id,
            "image_path": str(scene.image_path),
            "proposal_id": proposal["proposal_id"],
            "planner_name": proposal["planner"],
            "planner_config": proposal["planner_config"],
            "proposal": proposal,
            "valid_weed_target": valid_target,
            "approach_safety": approach_safety,
            "weed_removed": weed_removed,
            "crop_damaged": crop_damaged,
            "damage_severity": damage_severity,
            "annotator_confidence": confidence,
            "notes": notes.strip(),
        }
        feedback_id = save_feedback(db_path, payload)
        st.success(f"Saved feedback record #{feedback_id} to `{Path(db_path).resolve()}`")

with st.expander("Feedback database tools"):
    if st.button("Export all saved feedback to CSV"):
        export_path = Path("outputs/feedback_export.csv")
        rows = export_feedback_csv(db_path, export_path)
        st.success(f"Exported {rows} record(s) to `{export_path.resolve()}`")
    st.caption("Each submission is appended as a new record. Planner configuration and the complete proposal are retained as JSON.")
