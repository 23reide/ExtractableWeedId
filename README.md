# Weed Extraction Planning Platform (software-only prototype)

A small, modular benchmark for testing weed-extraction planning algorithms on agricultural images. The current implementation includes:

- a dataset adapter for the **CropAndWeed** dataset;
- a 2D geometric baseline using annotated plant bounding boxes and stem positions;
- JSONL action-proposal export;
- a Streamlit interface for human review/annotation;
- append-only feedback storage in SQLite, with CSV export;
- a tiny synthetic dataset so the workflow can be tested without downloading a research dataset.

> **Scope:** This prototype tests the *planner* against existing annotations. It does not yet run an off-the-shelf vision model on the raw image, physically simulate extraction, or validate that an action causes/avoids real damage. Those components are intentionally separate follow-on modules.

## 1. Dataset recommendation: CropAndWeed

Use the official dataset repository: <https://github.com/cropandweed/cropandweed-dataset>.

Why it fits this project:

- It contains real-world agricultural images and labels for crop and weed categories.
- The annotation CSVs contain individual plant boxes and stem coordinates (`Left, Top, Right, Bottom, Label ID, Stem X, Stem Y`), which are immediately useful for 2D approach planning.
- It also provides semantic label masks and recording-condition metadata for future segmentation-based experiments.
- The `CropsOrWeed9` variant groups plants into eight crop classes plus a weed class. This prototype uses IDs `0..7` as crops and `8` as weeds for that variant.

The dataset is released for **non-commercial purposes**. Review its license and cite the original WACV 2023 paper before using it in research or publishing results.

### Prepare the dataset

Follow the dataset maintainers' setup instructions. In a terminal:

```bash
git clone https://github.com/cropandweed/cropandweed-dataset.git
cd cropandweed-dataset/cnw
python -m pip install requests tqdm numpy pillow
python setup.py --data_root /absolute/path/to/cropandweed_data
```

The maintainers' setup script downloads and extracts the image/annotation archives and creates mapped annotation variants. Use the resulting directory passed to `--data_root` below; it should contain folders similar to:

```text
cropandweed_data/
├── images/
├── bboxes/
│   └── CropsOrWeed9/
└── labelIds/
    └── CropsOrWeed9/
```

If the upstream setup script or directory layout changes, adjust the dataset path/layout rather than copying images or annotations into the project source tree. This adapter searches recursively for matching per-image CSV files under `bboxes/`.

## 2. Install this project

From this project's directory:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Run the included synthetic demo

The demo data is artificial; it is included only to test the code path and UI.

```bash
python make_demo_data.py
python -m unittest discover -s tests -v
python run_planner.py --data-root ./demo_dataset --variant CropsOrWeed9 --max-images 2 --out outputs/demo_proposals.jsonl
streamlit run app.py
```

When the app opens, enter `demo_proposals.jsonl` in the **Proposal file in outputs** field. The app reads proposals from that file and loads the image and annotations referenced by each proposal. Generate a different proposal file with `run_planner.py` to review another dataset or planner configuration.

## 4. Export action proposals in batch

```bash
python run_planner.py \
  --data-root /absolute/path/to/cropandweed_data \
  --variant CropsOrWeed9 \
  --max-images 100 \
  --gripper-radius-px 8 \
  --safety-margin-px 3 \
  --approach-length-px 55 \
  --angle-step-deg 15 \
  --out outputs/proposals.jsonl
```

Each JSONL row represents one target weed, including the target base coordinates, selected approach angle, entry point, per-angle clearance estimates, planner configuration, and `attempt`/`skip` decision. All coordinates and distances are currently **pixels**. There is no camera calibration or physical scale.

## 5. Review and store feedback

Launch the app with `streamlit run app.py`. Enter a proposal JSONL filename from the `outputs/` folder, select an action, examine the proposed approach, and annotate:

- whether the selected plant is a valid weed target;
- whether the approach looks safe for neighboring crops;
- whether extraction appears plausible from the image;
- whether neighboring crop damage is likely;
- damage severity, confidence, and optional notes.

Each submission is appended to `outputs/feedback.sqlite`. The row retains the image path, complete proposal, selected planner, planner parameters, annotation labels, and timestamp. The **Export all saved feedback to CSV** tool writes `outputs/feedback_export.csv`.

Use `uncertain` / `cannot_tell` when a static image is insufficient. These are visual assessments, not real extraction outcome measurements.

## 6. Geometry baseline

For each annotated weed, the planner:

1. takes the annotated stem location as the 2D target (falls back to the bottom-centre of the plant box if unavailable);
2. samples approach directions around the target;
3. forms an approach segment from an entry point toward the target base;
4. samples points along the path and measures their distance to each crop's axis-aligned bounding box;
5. considers a path geometrically safe when its minimum estimated clearance is at least `gripper_radius_px + safety_margin_px` and the entry point is inside the image;
6. proposes the safe direction with the largest clearance, or recommends `skip` if none pass.

This is deliberately a simple, explainable baseline. Bounding boxes can be over-conservative, and the current circular-tool approximation does not model a particular end-effector shape or arm kinematics. Improve it later with per-plant masks, configuration-space geometry, or a collision-checked robot planner.

## 7. Architecture and intended extension points

- **Perception adapter:** convert raw model detections/segmentation into the common scene representation. The current adapter supplies ground-truth boxes/stem points from dataset annotations so planner evaluation is isolated from perception quality.
- **Planner interface:** replace `propose_for_scene` with geometry, pretrained image-model, or learned decision-layer implementations that emit the same action-proposal fields.
- **Feedback/training:** export the SQLite labels to train/evaluate a decision layer. Split train/test data by image (or recording session) to avoid leakage from multiple actions on the same scene.
- **Robot adapter:** convert an accepted image-space action into calibrated world coordinates and then robot-specific motion commands. Keep that separate from this planner.

## Citation

Please cite the dataset's original paper if you use CropAndWeed:

> Daniel Steininger, Andreas Trondl, Gerardus Croonen, Julia Simon, and Verena Widhalm. “The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation.” *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2023, pp. 3729–3738.

Dataset repository and license: <https://github.com/cropandweed/cropandweed-dataset>
