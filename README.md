# WeedBot — Weed Extraction Planning Platform (software-only prototype)

A modular, active-learning platform for testing weed-extraction planning algorithms on
agricultural images. The system identifies weeds, crops, and soil; proposes 2D extraction
coordinates; collects binary human feedback through a Streamlit UI; and fine-tunes a
YOLOv8-nano-seg model on that feedback via a Google Colab notebook.

```
Raw image
  → Perception backend (ground-truth CSV  or  YOLOv8-nano-seg)
  → Scene (weeds / crops / soil with bounding boxes + stem coords)
  → Geometry planner (clearance-based approach path selection)
  → Streamlit review UI (mask overlays + pull/no-pull decision)
  → Binary human feedback (correct / incorrect / uncertain)
  → SQLite feedback store
  → Export: image patches + labels.csv
  → Google Colab fine-tuning notebook → best.pt
  → Swap weights in config.yaml → improved perception
```

---

## 1. Dataset: CropAndWeed

The ground-truth backend uses the [CropAndWeed dataset](https://github.com/cropandweed/cropandweed-dataset).

```bash
git clone https://github.com/cropandweed/cropandweed-dataset.git
cd cropandweed-dataset/cnw
python -m pip install requests tqdm numpy pillow
python setup.py --data_root /absolute/path/to/cropandweed_data
```

The resulting directory should contain:

```
cropandweed_data/
├── images/
├── bboxes/
│   └── CropsOrWeed9/
└── labelIds/
    └── CropsOrWeed9/
```

Update `data_root` in `config.yaml` to point to this directory.

---

## 2. Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3. Quick demo (no dataset download required)

```powershell
python make_demo_data.py          # generates demo_dataset/
python -m unittest discover -s tests -v
streamlit run app.py
```

In the app sidebar, enter `demo_dataset/images/demo_0001.png` in the path field.

---

## 4. Configuration — config.yaml

All runtime settings live in `config.yaml` at the project root.

```yaml
# "groundtruth" uses dataset annotation CSVs (fast, no model required)
# "model" runs YOLOv8-nano-seg on the raw image
backend: groundtruth

# Path to fine-tuned weights (.pt file).
# Leave blank to use pretrained COCO weights (placeholder until you retrain).
model_weights_path: ""

# Dataset root for the ground-truth backend.
data_root: "demo_dataset"
dataset_variant: "CropsOrWeed9"

# Maps YOLOv8 class names → "crop" / "weed" / "soil".
# Update this after retraining to match your trained class names.
label_map:
  crop: crop
  weed: weed
  potted plant: crop
  grass: weed
```

---

## 5. Active learning loop

### Step 1 — Review images and collect feedback

```powershell
streamlit run app.py
```

Open an image, inspect the pull/skip decision, then select **correct**, **incorrect**, or
**uncertain** and click **Save feedback**.

### Step 2 — Export training data

Once you have enough labeled samples, open the **Active Learning / Retrain** expander in
the sidebar and click **Retrain now**. This:

- Crops 256 × 256 px patches around each labeled weed.
- Writes `outputs/training_export/labels.csv`.
- Zips everything to `outputs/training_export.zip`.
- Displays step-by-step upload instructions.

### Step 3 — Fine-tune in Google Colab

1. Upload `outputs/training_export.zip` to **Google Drive** → `WeedBot/`.
2. Open `notebooks/finetune_colab.ipynb` in Google Colab
   (Runtime → Change runtime type → T4 GPU for best speed).
3. Update `DRIVE_ZIP_PATH` in Cell 1 and run all cells.
4. Trained weights are saved to `WeedBot/weights/best.pt` on your Drive.

### Step 4 — Swap weights and reload

1. Download `best.pt` from Drive to (e.g.) `outputs/weights/best.pt`.
2. Edit `config.yaml`:
   ```yaml
   backend: model
   model_weights_path: outputs/weights/best.pt
   label_map:
     crop: crop
     weed: weed
   ```
3. Restart the app — it picks up the new weights automatically.

---

## 6. Notes on the feedback label design

The binary feedback ("correct" / "incorrect") conflates two independent signals into one:

- **Was this actually a weed?** — a ground-truth fact about the image.
- **Was the approach path safe?** — a judgment about the planner's geometry given the crop layout.

A SKIP proposal can be "incorrect" because the weed was real but surrounded by crops (planner was too conservative), or because the plant wasn't a weed at all. An ATTEMPT proposal can be "incorrect" because the model misidentified the plant, or because the approach angle was bad. The current schema does not distinguish between these, which is a reasonable simplification for early-stage data collection but may cause noisy training signal if the dataset grows large.

This somewhat complicates the decision to let annotations accumulate between model retrainings. A future improvement would be to add a `model_version` field to each feedback row (stamped from `config.yaml` at annotation time) so that labels from each training round can be identified and optionally filtered. This makes it possible to experiment with discarding annotations from early, low-quality model versions without losing the ability to reconstruct the full history.

---

## 7. Architecture and extension points

| Layer | Current implementation | How to extend |
|---|---|---|
| **Perception** | `GroundTruthBackend` (CSV annotations) or `ModelBackend` (YOLOv8-nano-seg) | Implement `PerceptionBackend.predict()` in `weedbot/perception.py` |
| **Planner** | `bbox_clearance_v1` geometric baseline | Replace `propose_for_scene` in `weedbot/geometry.py` |
| **Feedback storage** | SQLite via `weedbot/storage.py` | Add columns; migration runs automatically at connect |
| **Training export** | 256 × 256 px patches + `labels.csv` | Edit `weedbot/export_training_data.py` |
| **Fine-tuning** | Colab notebook (`notebooks/finetune_colab.ipynb`) | Swap model architecture; add augmentation |
| **Robot adapter** | Not implemented | Convert accepted image-space action to world coords + motion commands |

---

## 8. Tests

```powershell
python -m unittest discover -s tests -v
```

23 tests covering dataset loading, storage migration, perception interface, visualisation,
export pipeline, and geometry planner.

---

## Citation

If you use the CropAndWeed dataset, please cite:

> Daniel Steininger, Andreas Trondl, Gerardus Croonen, Julia Simon, and Verena Widhalm.
> "The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed
> Manipulation." *WACV 2023*, pp. 3729–3738.

Dataset repository and license: <https://github.com/cropandweed/cropandweed-dataset>
