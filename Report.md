# Handwritten Circuit Diagram Recognition — Report

Oct 21, 2026 ·

## Goal and dataset

A YOLO11m detector finds and classifies the components of hand-drawn circuits with **mAP50 0.989 and mAP50-95 0.810 on the held-out test set**. That means 98.6 % of the 1,894 test components are found with the correct class.

- **Task:** take an image of a hand-drawn circuit, find every component and say what it is.
- **Data:** Digitize-HCD, 1,277 scanned hand-drawn circuits, 17 component classes, COCO annotations (`component_annotations.json`).
- **Delivered:** two Kaggle notebooks (component detection, and the bonus value reading), all metrics and figures in the repository, and the trained weights in a GitHub Release.

## Method

Every decision was made on validation data; the test set was evaluated once, at the end.

1. **Data checks.** All images are present. Classes are imbalanced: in the test split, from 30 boxes (MOSFET-P) to 329 (Resistor). Boxes are clipped to the image, and boxes with a side under 5 px are removed.
2. **Split.** 80/10/10 **by image** (1,021 / 128 / 128), so no drawing appears in two splits. It is stratified on each image's rarest class, so every class appears in train, validation and test.
3. **Model choice.** YOLO11m, pretrained on COCO.
   - A one-stage detector fits the task: one class per box and many small objects per image, with no need for masks or region proposals.
   - Transfer learning suits a dataset of \~1k images.
   - The medium size keeps fine details (e.g. arrow direction in sources, the bar of a Zener diode) and trains in about 2 h on a free Kaggle T4.
   - Other architectures were not benchmarked.
4. **Training settings.**
   - `imgsz = 1024`: the scans are large, and at the default 640 px many symbols become only a few pixels wide.
   - Up to 100 epochs, early stopping with patience 20, batch 4.
   - Horizontal flips are disabled, because mirrored symbols and handwriting do not occur. A ±5° rotation imitates slightly tilted scans.
   - Training stopped at epoch 97; the best epoch was 77 (1.9 h).
5. **Evaluation.**
   - mAP50 and mAP50-95 at `conf = 0.001`, so the full precision-recall curve is used.
   - The operating threshold (**0.78**) is the best mean F1 on validation, and is reused unchanged on test.
   - The confusion matrix is computed at that threshold, and drives the error analysis.

## Results

Test and validation scores are almost identical, so the threshold and other choices made on validation did not overfit it.

| Split | Images | Components | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Validation | 128 | 1,914 | 0.987 | 0.808 | 0.983 | 0.984 |
| **Test** | 128 | 1,894 | **0.989** | **0.810** | 0.986 | 0.985 |

At the 0.78 threshold on test, 1,868 of 1,894 components are found with the correct class. The 29 errors break down as follows:

| Error type | Count | Main cases |
| --- | --- | --- |
| Missed | 19 | Wire Crossover (8), GND (6), Inductor (3) |
| Wrong class | 7 | BJT PNP→NPN, MOSFET P→N, Diode→Zener, V-AC→I-AC, V-DC→I-DC, MOSFET P→BJT PNP, Inductor→Resistor |
| False alarm | 3 | GND (2), Zener Diode (1) |

- **Wrong classes are look-alike pairs.** Each pair differs by one small detail: the arrow direction in a transistor, the bar of a Zener diode, or the sign inside a source.
- **Misses are small or ambiguous symbols.** A wire crossover is a small hop on a line, and GND symbols are small.
- **mAP50 is almost 1 but mAP50-95 is 0.81.** The model finds the components, but the box edges are less exact. Hand-drawn symbols have no well-defined edge, and annotators placed them differently. The lowest mAP50-95 values are V-DC (one port) 0.70, GND 0.74 and Diode 0.74.

## Bonus: reading the values

On test, **90.0 % of the written values are found, read exactly and attached to a component of a compatible class**. Whether it is the right component cannot be measured (see below). Almost every text is found (99.7 %), so reading is the bottleneck.

1. **Text detector:** YOLO11s with a single class `text`, same split and recipe. A small model is enough for one class without fine distinctions.
2. **Reader:** TrOCR (`microsoft/trocr-base-handwritten`), fine-tuned for 3 epochs on \~9,500 training crops; the best epoch on validation (epoch 3) is kept. Without fine-tuning it reads only 10.9 % of the validation values exactly, mostly because it misreads `μ` and `Ω`.
3. **Matching:** the unit restricts the possible classes (`Ω` → resistor, `F` → capacitor, `H` → inductor, `Hz` → AC sources, `V` → voltage sources, `A` → current sources). The text then goes to the nearest allowed component. A value without a unit takes it from the class (`1k` next to a resistor → `1kΩ`).

| Stage | Validation | Test |
| --- | --- | --- |
| Text detector, mAP50 / mAP50-95 | 0.995 / 0.786 | 0.995 / 0.793 |
| Reader on true text boxes, exact match | 0.916 | 0.911 |
| Reader on true text boxes, character error rate | 0.039 | 0.050 |
| End to end: found and read exactly | 0.902 | 0.901 |
| End to end: found, read and attached | 0.898 | 0.900 |

The dataset has no text-to-component links, so matching is checked only indirectly. On the true boxes, distance alone picks a component whose class fits the unit 94.9 % of the time, and the unit rule changes the choice in 5.1 % of cases.

## Alternatives considered

Each choice was weighed against simpler or stronger options. The ones below were considered but not run, for the reason given.

| Alternative | Trade-off | Decision |
| --- | --- | --- |
| **Task 1** |  |  |
| Two-stage detector (Faster R-CNN) | Often more precise on boxes, but slower to train and run. | Not needed here: YOLO11m reaches mAP50 0.989, with 29 errors out of 1,894 test components, and trains in about 2 h on a free T4. |
| Smaller or larger YOLO (11s / 11l) | Faster to train vs. more capacity for fine details. | 11m as a middle ground for a free T4; a size sweep is the first next step. |
| Default 640 px input | Faster training. | Rejected: the box-size check showed many symbols shrink to a few pixels at 640. |
| Tiled inference (SAHI) for small symbols | Better on tiny objects, but more complex and slower. | Not needed at 1024 px; worth testing for wire crossovers, the most-missed class. |
| Class weights or oversampling for rare classes | Could help the rarest classes. | Not needed: rare classes such as MOSFET-P (30 test boxes) still reach AP50 0.96. |
| Split by writer | Stricter test of generalisation. | Not possible: the dataset has no writer id. |
| **Bonus** |  |  |
| Text as an 18th class in the main YOLO | One model instead of two. | Would require retraining the main detector, so a separate text detector was trained instead. |
| Match by distance only, without the unit rule | Slightly simpler. | Kept the unit rule: it is a few lines and changes the choice in 5.1 % of cases. |
| Off-the-shelf OCR (EasyOCR, PaddleOCR) without training | No training needed. | Likely to fail on handwritten `Ω` and `μ`, as zero-shot TrOCR did (10.9 % exact match); fine-tuning was needed. |

## Limitations and next steps

The main limit is that every result comes from clean scans of one dataset; real phone photos were not evaluated.

- **Domain shift.** On one hand-drawn phone photo outside the dataset, the 0.78 threshold was too strict, and some components were only found at 0.25. Photo-style augmentation (blur, lighting, perspective) or a few labelled photos would help.
- **Split by image, not by writer.** The same handwriting may appear in train and test, so scores on unseen writers may be lower.
- **One configuration.** There was one model per task and one seed, with no comparison with other architectures or sizes. A small sweep (YOLO11s/m/l, 640 vs 1024 px) would justify the choice with numbers.
- **Look-alike classes.** Most class errors are look-alike pairs (PNP/NPN, P/N MOSFET, voltage/current source). Higher resolution or class-specific crops could reduce them.
- **Bonus.** The reader only knows the dataset's labelling style (no designators such as `R1`). Its remaining errors are unit prefixes (m/M/μ), repeated digits (77V for 7V) and formula formats. Matching correctness is not measured per instance.

The project used Claude (an LLM) during development. Its prompts and responses are in `LLM_USAGE.md`.
