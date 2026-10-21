# Handwritten circuit diagram recognition

A computer vision pipeline for detecting electrical components in hand-drawn circuit diagrams and reading their handwritten values.

Using the [Digitize-HCD dataset](https://data.mendeley.com/datasets/rngcz5wtv8/1), the project combines YOLO-based object detection, fine-tuned TrOCR and unit-aware matching to turn scanned circuit diagrams into annotated outputs.

![Components in green, values in red with a line to their component](task2_value_reading/results/figures/example_values.png)

## Results (test set)

| Task | Approach | Result |
|---|---|---|
| **1: component detection** | YOLO11m at 1024 px | mAP50 **0.989**, mAP50-95 **0.810**; 98.6 % of components found with the correct class |
| **2 (bonus): value reading** | YOLO11s text detector + fine-tuned TrOCR + unit-aware matching | **90.0 %** of values found, read exactly and attached to a compatible component |

Method, error analysis and design choices are in [`Report.md`](Report.md).

## Quick start

Download `submission_outputs.zip` from the [Release](../../releases), then:

```python
from ultralytics import YOLO

model = YOLO("yolo11m_1024/weights/best.pt")
model.predict("path/to/your_circuit.png", imgsz=1024, conf=0.78, save=True)
```

## Reproduce

Run on [Kaggle](https://www.kaggle.com/) (one T4 GPU) with the [Digitize-HCD dataset](https://www.kaggle.com/datasets/diegofogued/digitize-hcd) attached:

1. **Task 1:** [`circuit-detector`](https://www.kaggle.com/code/diegofogued/circuit-detector) (~2 h). Outputs `submission_outputs.zip`.
2. **Task 2:** upload that zip as a Kaggle dataset, attach it together with [Digitize-HCD dataset](https://www.kaggle.com/datasets/diegofogued/digitize-hcd), then run [`value-reading`](https://www.kaggle.com/code/diegofogued/value-reading) (~1 h).

Set `RUN_TEST = True` only for the final run; the test set is evaluated once.

## Contents

- `task1_detection/`, `task2_value_reading/`: notebooks and saved results (metrics, logs, figures).
- [Release](../../releases): trained detector weights and all run outputs. The fine-tuned TrOCR (~1.3 GB) is not included; the Task 2 notebook retrains it in ~22 min.
- [`LLM_USAGE.md`](LLM_USAGE.md): prompts and responses from the LLM (Claude) used during the task.

## Licences

Dataset: CC BY 4.0 ([Ahmed et al., *Data in Brief*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11808513/)). Ultralytics YOLO: AGPL-3.0. TrOCR: see the `microsoft/trocr-base-handwritten` model card.
