"""WeedBot — Streamlit review and active-learning UI.

Flow
----
1.  Point to a folder of images in the sidebar.
2.  The app queues all images in that folder and loads them one at a time.
3.  The configured perception backend detects weeds/crops/soil.
4.  The geometry planner proposes extraction coordinates for each weed.
5.  Image is shown on the left; proposal details + feedback form on the right.
6.  Submitting feedback automatically advances to the next image.
7.  Keyboard hint: Tab between radio options, Enter to submit the form.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import streamlit as st

from weedbot.export_training_data import export_for_training, zip_training_export
from weedbot.geometry import PlannerConfig, propose_for_scene
from weedbot.perception import PerceptionResult, backend_from_config, load_config
from weedbot.storage import count_labeled_feedback, export_feedback_csv, save_feedback
from weedbot.visualize import draw_scene

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WeedBot Review", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUTS_DIR / "feedback.sqlite"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Load config + backend (cached — doesn't reload on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading perception backend…")
def _get_backend():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    return backend_from_config(cfg), cfg


try:
    backend, cfg = _get_backend()
except Exception as exc:
    st.error(f"Failed to load perception backend: {exc}")
    st.code(traceback.format_exc())
    st.stop()

backend_name = str(cfg.get("backend", "groundtruth")).strip().lower()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "queue" not in st.session_state:
    st.session_state.queue: list[Path] = []
if "queue_index" not in st.session_state:
    st.session_state.queue_index: int = 0
if "folder_input" not in st.session_state:
    st.session_state.folder_input: str = ""


def _advance() -> None:
    """Move to the next image in the queue."""
    st.session_state.queue_index += 1


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂 Image folder")
    folder_input = st.text_input(
        "Folder path (relative or absolute)",
        value=st.session_state.folder_input,
        placeholder="demo_dataset/images",
    )

    if folder_input.strip() != st.session_state.folder_input:
        # Folder changed — rebuild the queue and reset position.
        st.session_state.folder_input = folder_input.strip()
        candidate = Path(folder_input.strip())
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_dir():
            found = sorted(
                p for p in candidate.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            st.session_state.queue = found
            st.session_state.queue_index = 0
        else:
            st.session_state.queue = []
            st.session_state.queue_index = 0

    queue: list[Path] = st.session_state.queue
    idx: int = st.session_state.queue_index

    if queue:
        remaining = max(0, len(queue) - idx)
        st.caption(f"**{remaining}** image(s) remaining of **{len(queue)}** total")
        # Manual jump — lets the user skip around if needed.
        jump = st.number_input(
            "Jump to image #", min_value=1, max_value=len(queue),
            value=min(idx + 1, len(queue)), step=1,
        )
        if jump - 1 != idx:
            st.session_state.queue_index = jump - 1
            st.rerun()
    elif folder_input.strip():
        st.warning("No images found in that folder.")

    st.divider()
    st.header("⚙️ Backend")
    st.markdown(
        f"**Active:** `{backend_name}`  \n"
        f"**Weights:** `{cfg.get('model_weights_path') or 'pretrained COCO'}`  \n"
        "Edit `config.yaml` to change."
    )

    st.divider()
    with st.expander("🔁 Active Learning / Retrain", expanded=False):
        labeled_count = count_labeled_feedback(DB_PATH)
        st.metric("Labeled samples", labeled_count)

        retrain_log_path = OUTPUTS_DIR / "retrain_log.json"
        if retrain_log_path.exists():
            try:
                log = json.loads(retrain_log_path.read_text(encoding="utf-8"))
                st.caption(
                    f"Last export: **{log.get('exported_at', '—')}** "
                    f"({log.get('sample_count', '?')} samples)"
                )
            except Exception:
                pass

        if st.button("📦 Retrain now", disabled=labeled_count == 0):
            export_dir = OUTPUTS_DIR / "training_export"
            with st.spinner("Exporting…"):
                try:
                    import shutil
                    n = export_for_training(DB_PATH, export_dir)
                    zip_path = zip_training_export(
                        export_dir, OUTPUTS_DIR / "training_export.zip"
                    )
                    shutil.copy2(export_dir / "retrain_log.json", retrain_log_path)
                    st.success(f"Exported **{n}** sample(s) → `{zip_path.name}`")
                    st.markdown(
                        "**Next steps:**\n"
                        f"1. Upload `{zip_path.name}` to Google Drive → `WeedBot/`\n"
                        "2. Open `notebooks/finetune_colab.ipynb` in Colab\n"
                        "3. Update `DRIVE_ZIP_PATH` and run all cells\n"
                        "4. Download `best.pt`, place at `outputs/weights/best.pt`\n"
                        "5. Set `backend: model` and `model_weights_path` in `config.yaml`\n"
                        "6. Restart the app"
                    )
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

        if labeled_count > 0 and st.button("📊 Export feedback CSV"):
            csv_path = OUTPUTS_DIR / "feedback_export.csv"
            n_rows = export_feedback_csv(DB_PATH, csv_path)
            st.success(f"Exported {n_rows} row(s)")

# ---------------------------------------------------------------------------
# Resolve current image from queue
# ---------------------------------------------------------------------------
queue = st.session_state.queue
idx = st.session_state.queue_index

if not queue:
    st.title("🌱 WeedBot — Weed Extraction Review")
    st.info(
        "Enter a folder path in the sidebar to begin.  \n"
        "Example: `demo_dataset/images`  \n\n"
        "Run `python make_demo_data.py` first if you need a demo dataset."
    )
    st.stop()

if idx >= len(queue):
    st.title("🌱 WeedBot — Weed Extraction Review")
    st.success(
        f"✅ All {len(queue)} image(s) in the queue have been reviewed.  \n"
        "Enter a new folder path or adjust the jump control to go back."
    )
    st.stop()

image_path = queue[idx]

st.title(
    f"🌱 WeedBot  ·  {image_path.name}  ·  "
    f"{idx + 1} / {len(queue)}"
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: save a feedback payload. Returns the feedback_id.
# Advancing (proposal or image) is handled by the caller so each code path
# can choose what "next" means without the helper making that decision.
# ---------------------------------------------------------------------------
def _submit_feedback(
    image_id: str,
    image_path_str: str,
    proposal_id: str,
    planner_name: str,
    planner_config: dict,
    proposal: dict,
    pull_correct: str,
    notes: str,
) -> int:
    payload = {
        "image_id": image_id,
        "image_path": image_path_str,
        "proposal_id": proposal_id,
        "planner_name": planner_name,
        "planner_config": planner_config,
        "proposal": proposal,
        "pull_decision_correct": pull_correct,
        "notes": notes,
    }
    fid = save_feedback(DB_PATH, payload)
    total = count_labeled_feedback(DB_PATH)
    st.success(f"Saved #{fid} · {total} labeled sample(s) total")
    return fid


# ---------------------------------------------------------------------------
# Run perception backend
# ---------------------------------------------------------------------------
with st.spinner("Running perception…"):
    try:
        perception_result: PerceptionResult = backend.predict(image_path)
    except FileNotFoundError as exc:
        st.error(
            f"Annotation file not found for `{image_path.name}`: {exc}  \n\n"
            "With the **groundtruth** backend, every image needs a matching "
            "`bboxes/<variant>/<stem>.csv` file. "
            "Check that `data_root` in `config.yaml` is correct."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Perception error: {exc}")
        st.code(traceback.format_exc())
        st.stop()

scene = perception_result.scene
masks = perception_result.masks

# ---------------------------------------------------------------------------
# Branch 1: no plants detected at all
# ---------------------------------------------------------------------------
if not scene.plants:
    left_col, right_col = st.columns([1.6, 1])
    with left_col:
        st.image(str(image_path), caption="No detections", use_container_width=True)
    with right_col:
        st.subheader("No plants detected")
        st.markdown("The model found nothing in this image.")
        with st.form(key=f"no-plants-{image_path.stem}-{idx}"):
            pull_correct = st.radio(
                "Was that correct?",
                options=["correct", "incorrect", "uncertain"],
                horizontal=True,
                index=2,
                help="**C**orrect · **I**ncorrect · **U**ncertain",
            )
            notes = st.text_area("Notes (optional)", height=68)
            if st.form_submit_button("💾 Save & next image →", type="primary"):
                _submit_feedback(
                    image_path.stem, str(image_path),
                    f"{image_path.stem}::no_detections::missed",
                    "none", {},
                    {"decision": "skip", "reason": "no_plants_detected",
                     "target_xy": [0, 0], "entry_xy": [0, 0],
                     "approach_angle_deg": 0, "safe_candidate_count": 0,
                     "candidate_approaches": [], "planner": "none", "planner_config": {}},
                    pull_correct, notes.strip(),
                )
                _advance()
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Run geometry planner
# ---------------------------------------------------------------------------
planner_config = PlannerConfig()
proposals = propose_for_scene(scene, planner_config)

# ---------------------------------------------------------------------------
# Branch 2: plants found but no weeds
# ---------------------------------------------------------------------------
if not proposals:
    left_col, right_col = st.columns([1.6, 1])
    with left_col:
        st.image(
            draw_scene(scene, masks=masks),
            caption="Green = crop · Yellow dot = stem",
            use_container_width=True,
        )
    with right_col:
        st.subheader("No weeds detected")
        st.markdown(
            f"Found **{len(scene.crops)}** crop(s), **0** weeds — nothing to plan."
        )
        with st.form(key=f"no-weeds-{scene.image_id}-{idx}"):
            pull_correct = st.radio(
                "Was that correct?",
                options=["correct", "incorrect", "uncertain"],
                horizontal=True,
                index=2,
                help="**C**orrect · **I**ncorrect · **U**ncertain",
            )
            notes = st.text_area("Notes (optional)", height=68)
            if st.form_submit_button("💾 Save & next image →", type="primary"):
                _submit_feedback(
                    scene.image_id, str(scene.image_path),
                    f"{scene.image_id}::no_weeds_detected::missed",
                    "none", {},
                    {"decision": "skip", "reason": "no_weeds_detected",
                     "target_xy": [0, 0], "entry_xy": [0, 0],
                     "approach_angle_deg": 0, "safe_candidate_count": 0,
                     "candidate_approaches": [], "planner": "none", "planner_config": {}},
                    pull_correct, notes.strip(),
                )
                _advance()
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Branch 3: normal flow — proposals exist
# ---------------------------------------------------------------------------
st.caption(
    f"Detected: **{len(scene.weeds)}** weed(s) · **{len(scene.crops)}** crop(s) · "
    f"**{len(proposals)}** proposal(s)"
)

# If there are multiple proposals on one image, let the user step through them.
# We track which proposal index we're on in session_state, resetting when the
# image changes.
if "proposal_index" not in st.session_state:
    st.session_state.proposal_index = 0
# Reset proposal index when the image changes.
if st.session_state.get("_last_image_idx") != idx:
    st.session_state.proposal_index = 0
    st.session_state["_last_image_idx"] = idx

p_idx = min(st.session_state.proposal_index, len(proposals) - 1)
proposal = proposals[p_idx]

if len(proposals) > 1:
    st.caption(f"Proposal **{p_idx + 1}** of **{len(proposals)}** for this image")

# ---------------------------------------------------------------------------
# Two-column layout: image left, details + feedback right
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.6, 1])

with left_col:
    img = draw_scene(scene, selected_proposal=proposal, masks=masks)
    caption_parts = ["Green = crop · Red = weed · Yellow dot = stem · Blue = approach"]
    if masks:
        caption_parts.append("Coloured fills = model masks")
    st.image(img, caption=" · ".join(caption_parts), use_container_width=True)
    if masks:
        st.image(str(image_path), use_container_width=True)

with right_col:
    # --- Proposal summary ---
    decision_icon = "✅" if proposal["decision"] == "attempt" else "⏭"
    st.markdown(
        f"**Decision:** {decision_icon} `{proposal['decision'].upper()}`  \n"
        f"**Target:** `{proposal['target_weed_id'].rsplit(':', 1)[-1]}`  \n"
        f"**Target XY:** `{proposal['target_xy']}`  \n"
        f"**Approach:** `{proposal['approach_angle_deg']}°` · "
        f"`{proposal['safe_candidate_count']}` safe direction(s)"
    )
    if proposal.get("nearest_crop_bbox_distance_px") is not None:
        st.markdown(f"**Nearest crop:** `{proposal['nearest_crop_bbox_distance_px']} px`")

    with st.expander("Candidate approaches", expanded=False):
        st.dataframe(
            [
                {
                    "angle°": c["angle_deg"],
                    "entry_x": round(c["entry_xy"][0], 1),
                    "entry_y": round(c["entry_xy"][1], 1),
                    "clearance": c["min_crop_clearance_px"],
                    "safe": "✅" if c["geometric_safe"] else "❌",
                }
                for c in proposal["candidate_approaches"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- Feedback form ---
    st.markdown("**Was this decision correct?**")

    # Determine what happens after submit: if more proposals remain on this
    # image, advance to the next proposal; otherwise advance to the next image.
    more_proposals_remain = p_idx < len(proposals) - 1

    with st.form(key=f"feedback-{proposal['proposal_id']}-{idx}"):
        pull_correct = st.radio(
            f"Planner said **{proposal['decision'].upper()}** — correct?",
            options=["correct", "incorrect", "uncertain"],
            horizontal=True,
            index=2,
        )
        notes = st.text_area("Notes (optional)", height=68)

        next_label = (
            "💾 Save & next proposal →" if more_proposals_remain
            else "💾 Save & next image →"
        )
        submitted = st.form_submit_button(next_label, type="primary")

        if submitted:
            try:
                _submit_feedback(
                    scene.image_id,
                    str(scene.image_path),
                    proposal["proposal_id"],
                    proposal["planner"],
                    proposal["planner_config"],
                    proposal,
                    pull_correct,
                    notes.strip(),
                )
                if more_proposals_remain:
                    # Stay on this image, move to the next proposal.
                    st.session_state.proposal_index = p_idx + 1
                else:
                    # All proposals on this image are done — move to next image.
                    _advance()
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save: {exc}")
                st.code(traceback.format_exc())
