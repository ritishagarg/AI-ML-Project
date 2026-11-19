# AI-ML-Project

Plant leaf disease detection with **YOLOv8**, plus an evaluation of whether **SAHI** (Slicing Aided Hyper Inference) improves detection of small / densely packed objects over the plain YOLOv8 baseline.

The working dataset, trained weights and all experiment outputs live in the [`YOLO/`](YOLO) folder.

---

## Contents

| File | What it is |
| --- | --- |
| [`YOLOv8_training.ipynb`](YOLOv8_training.ipynb) | **Training notebook.** Loads the dataset, trains `yolov8l.pt` for 50 epochs, validates the best checkpoint and runs inference (prediction) on the test images. |
| [`YOLO+SAHI_comparison.ipynb`](YOLO+SAHI_comparison.ipynb) | **Comparison notebook.** Runs the YOLOv8 baseline and the YOLOv8 + SAHI variants side by side: precision / recall / mAP tables, TP / FP / FN analysis on the labelled test set, and side-by-side bounding-box visualisations. |
| `YOLO/` | Dataset, labels, trained weights, run logs, metrics and result images (see [repository layout](#repository-layout)). |
| `.gitattributes` | Marks `*.pt` model weights as Git LFS objects. |

---

## Dataset

Taken from Roboflow: **Plants Diseases Detection and Classification — v12** (<https://universe.roboflow.com/graduation-project-2023/plants-diseases-detection-and-classification>), licensed CC BY 4.0. The export we use contains 2,509 leaf images (the Roboflow card lists 2,516) annotated in YOLO detection format across **29 classes** (Apple scab, Apple rust, Bell pepper spot, Corn blight/rust, Potato early & late blight, Tomato blights, mosaic and yellow-leaf virus, grape black rot, and more — the full list is in `YOLO/data.yaml`).

Annotation format is the standard YOLO box format (`class x_center y_center width height`, normalised 0–1); all images were exported at 800×800.

| Split | Images | Labels |
| --- | --- | --- |
| `train` | 2,010 | 2,010 |
| `valid` | 250 | 250 |
| `test` | 249 | 249 |

---

## Running the notebooks (Google Colab)

Both notebooks were written for Colab with a GPU. The paths in them assume the dataset sits at **`/content/drive/MyDrive/YOLO`** in Google Drive, so do the one-time setup below first.

### 1. Put the `YOLO` folder in Drive

The `YOLO` folder is tracked with Git LFS (the three `.pt` weights are ~251 MB of the ~501 MB total), so download it with [GitHub Desktop](https://desktop.github.com) or the Git CLI rather than the plain "Download ZIP" button — a ZIP from the web UI contains LFS *pointer files* instead of the real weights.

```bash
git lfs install
git clone https://github.com/ritishagarg/AI-ML-Project.git
```

Then upload the `YOLO` folder from the clone into the **root of your Google Drive** (`My Drive/YOLO`). The notebooks expect this exact layout:

```text
My Drive/
└── YOLO/
    ├── data.yaml
    ├── yolov8l.pt
    ├── train/{images,labels}
    ├── valid/{images,labels}
    ├── test/{images,labels}
    ├── runs/detect/train-4/weights/{best.pt,last.pt}
    └── external_results/
```

> Uploading 500 MB through the Drive web UI is slow. It's much faster to run `drive.mount(...)` in a Colab cell and copy the folder from a local path, or use the Google Drive desktop sync client.

### 2. Open the notebook in Colab

In Colab choose **File → Upload notebook** and pick `YOLOv8_training.ipynb` (do the same for `YOLO+SAHI_comparison.ipynb`), or open it straight from GitHub with **File → Open notebook → GitHub** and enter `ritishagarg/AI-ML-Project`.

### 3. Select a GPU runtime

**Runtime → Change runtime type → Hardware accelerator → T4 GPU** (training on CPU is very slow).

### 4. Run all cells

**Runtime → Run all** (or `Ctrl+F9`). The cells run in order:

1. `!nvidia-smi` — confirms the GPU is available.
2. `!pip install ultralytics` — installs the YOLO library.
3. `drive.mount('/content/drive')` — mounts Drive; click through the Google account authorisation prompt.
4. `%cd /content/drive/MyDrive/YOLO` — moves into the dataset folder.
5. `ultralytics.checks()` — prints the environment / CUDA status.
6. `%cd .../YOLO/data` then the `yolo ... mode=train` command — **trains** the model (50 epochs, `imgsz=224`, batch 16, `yolov8l.pt`). A T4 GPU takes roughly 1–2 hours; the best and last checkpoints plus all curves are written under `runs/detect/train*/`.
7. Result-image cells (`results.png`, `train_batch0.jpg`, `val_batch0_pred.jpg`) — display training curves and prediction samples.
8. `yolo ... mode=val` — **validates** `best.pt` on the validation split and prints Precision / Recall / mAP.
9. `yolo ... mode=predict ... source=.../test/images` — **inference on the test set**; annotated images are written to `runs/detect/predict/`.
10. The final cell displays the first 20 test predictions inline.

> The notebook's training and validation cells point at `runs/detect/train2/...`. If your run lands in a different folder (Ultralytics increments the name each time you train), update those paths — check `!ls runs/detect` after training.

### 5. Run the SAHI comparison notebook

`YOLO+SAHI_comparison.ipynb` uses the checkpoint from training (`YOLO/runs/detect/train-4/weights/best.pt`, already included in the repo). Its first cell installs `sahi ultralytics opencv-python`, then it mounts Drive. Run all cells again, but **edit the two image paths before running**:

- The visual-comparison cell expects your own image at `IMAGE_PATH` (the placeholder is a screenshot path such as `/content/Screenshot ....png`). Point it at any leaf image.
- The "Slices made by SAHI" cell also hard-codes a screenshot path; swap in the same image.

The metrics cells (TP/FP/FN over `test/images` + `test/labels`, and the mAP tables) need no edits — they read directly from the `YOLO` folder and write their CSVs to the notebook's working directory and `YOLO/external_results/`.

---

## Repository layout

```text
AI-ML-Project/
├── README.md
├── YOLOv8_training.ipynb            # training + validation + test inference
├── YOLO+SAHI_comparison.ipynb       # YOLO vs YOLO+SAHI evaluation
├── .gitattributes                   # Git LFS rules for *.pt
└── YOLO/
    ├── data.yaml                    # dataset config: 3 splits, 29 class names
    ├── README.dataset.txt           # Roboflow dataset card
    ├── README.roboflow.txt          # Roboflow export notes
    ├── yolov8l.pt                   # COCO-pretrained starting weights (LFS)
    ├── comparison.csv               # headline YOLOv8 vs YOLOv8+SAHI metrics
    ├── sahi_variants_comparison.csv # SAHI slice-size / NMS variants, all + small + large objects
    ├── subset_comparison.csv        # large-object vs small/multi-object subsets
    ├── sahi_small_objects_bootstrap.csv # SAHI variants on small objects with 95% CIs
    ├── train/{images,labels}        # 2,010 training images + YOLO labels
    ├── valid/{images,labels}        # 250 validation images + labels
    ├── test/{images,labels}         # 249 test images + labels
    ├── runs/detect/
    │   ├── train*/args.yaml         # training hyper-parameter logs (4 runs)
    │   ├── train-4/                 # final run: results.csv, curves, confusion matrices,
    │   │   └── weights/             #   batch/validation sample images, best.pt + last.pt (LFS)
    │   └── val*/                    # validation curves and prediction samples (9 runs)
    └── external_results/            # side-by-side YOLO vs SAHI image, per-class + summary CSVs
```

---

## Results at a glance

From `YOLO/comparison.csv` (full test split, confidence 0.25):

| Model | Precision | Recall | mAP50 | mAP50-95 | ms / image |
| --- | --- | --- | --- | --- | --- |
| YOLOv8 (baseline) | 0.522 | 0.480 | 0.510 | 0.352 | 1699 |
| YOLOv8 + SAHI | 0.342 | 0.559 | 0.481 | 0.301 | 189 |

SAHI trades precision for recall and is much cheaper per image, but on the whole test set it did **not** beat the baseline on mAP. The more careful study in `sahi_variants_comparison.csv` and `sahi_small_objects_bootstrap.csv` breaks results down by object size: SAHI's advantage appears mainly on *small / multi-object* images, where the mAP50 gain is positive but its 95% confidence interval (±0.04–0.08) still straddles zero, so the effect is not statistically conclusive at this dataset size. The baseline remains stronger on large-object images.

---

## Notes

- **Model weights** (`YOLO/yolov8l.pt`, `YOLO/runs/detect/train-4/weights/best.pt` and `last.pt`) are stored with Git LFS. Run `git lfs install` before cloning; otherwise you'll get ~130-byte pointer files.
- Training is CUDA-dependent in the notebooks. On CPU it will run but take many hours.
- SAHI results are sensitive to slice size and overlap; the notebooks expose `SLICE` and `OVERLAP` at the top of each cell so you can sweep them.
