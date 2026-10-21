# Handwritten circuit component detection

Detect and classify every component in hand-drawn circuit diagrams (Digitize-HCD dataset, 17 classes) with a YOLO11m model fine-tuned at 1024 px.

## Results (test set, evaluated once)

| Split | Images | Boxes | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| Validation | 128 | 1914 | 0.987 | 0.808 | 0.983 | 0.984 |
| **Test** | **128** | **1894** | **0.989** | **0.810** | **0.986** | **0.985** |

- Confidence threshold: **0.78**, chosen as the best mean F1 on validation and reused unchanged on test.
- At that threshold, out of 1894 test components, **19 were missed**, **7 got the wrong class** and there were **3 false alarms**. That is about 98.6 % correctly detected and classified.
- The class errors are between visually similar symbols: BJT-NPN vs BJT-PNP, MOSFET-N vs MOSFET-P, Diode vs Zener Diode, V-AC vs I-AC and V-DC vs I-DC. Most misses are *Wire Crossover* (8) and *GND* (6).
- mAP50 is close to 1, so almost every component is found. mAP50-95 (≈0.81) is lower because box edges on hand-drawn symbols are ambiguous. The weakest classes on that metric are *V-DC (one port)* and *GND* (≈0.70-0.74).
- Validation and test give nearly identical numbers, so the threshold and model choices did not overfit the validation set.

Per-class metrics: [`val_per_class_metrics.csv`](task1_detection/results/val_per_class_metrics.csv), [`test_per_class_metrics.csv`](task1_detection/results/test_per_class_metrics.csv).

| Training curves | Test confusion matrix |
|---|---|
| ![](task1_detection/results/figures/training_curves.png) | ![](task1_detection/results/figures/test_confusion_matrix.png) |

## Approach

1. **Data inspection:** COCO annotations, class counts (imbalanced) and box sizes after resizing, which motivated `imgsz=1024`.
2. **Cleaning and split:** boxes clipped to the image and degenerate boxes (< 5 px) removed. 80/10/10 split **by image** (no drawing appears in two splits), stratified on each image's rarest class. 1277 images: 1021 / 128 / 128.
3. **Training:** `yolo11m.pt` (COCO pretrained), 1024 px, batch 4, up to 100 epochs with early stopping (patience 20). Horizontal flips disabled (mirrored symbols don't occur), ±5° rotation. Training stopped at epoch 97; the best epoch was 77 (~1.9 h on a Tesla T4).
4. **Evaluation:** mAP on validation, threshold selection on validation, then a single evaluation on test (guarded by `RUN_TEST` and `TEST_DONE.txt`), followed by error analysis.

## Repository contents

```
task1_detection/
├── circuit-detector.ipynb   # full pipeline (shared without outputs)
└── results/
    ├── results_summary.json         # overall val/test metrics + threshold
    ├── val_per_class_metrics.csv
    ├── test_per_class_metrics.csv
    ├── split.csv                    # image_id -> train/val/test
    ├── training_log.csv             # per-epoch losses and metrics
    ├── train_args.yaml              # exact training configuration
    ├── requirements.txt             # package versions of the run
    ├── TEST_DONE.txt                # timestamp of the single test evaluation
    └── figures/                     # curves, confusion matrices, sample predictions
```

**Trained weights and all run outputs** are in `submission_outputs.zip` in the [Release](../../releases) (weights at `yolo11m_1024/weights/best.pt`).

## How to reproduce

The notebook is shared without outputs. All results above come from one full run on Kaggle and are stored in `results/`.

1. Create a Kaggle notebook, import `circuit-detector.ipynb` and attach the **Digitize-HCD** dataset (paths are found automatically).
2. Settings: Accelerator = GPU T4, Internet = On.
3. Set `RUN_TEST = True` and run all cells.

Retraining gives close but not bit-identical numbers, because GPU training is not fully deterministic. To check the reported metrics exactly, use the `best.pt` from the Release.
