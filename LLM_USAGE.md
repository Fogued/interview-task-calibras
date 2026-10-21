# LLM usage

I used Claude Opus 5.5 throughout the task, not only to write code: to research options, explore the data, understand concepts I had not used before and check my own reasoning. All training, evaluation and final decisions were run and checked by me on a Kaggle notebook, the LLM did not produce any of the reported numbers.

## How I used it

**1. Research: choosing the approach**

- Comparing detector families for this problem (one-stage YOLO vs. two-stage Faster R-CNN) and model sizes (YOLO11s / m / l) for a ~1k-image dataset on a free T4.
- For the bonus: options for reading handwritten values (off-the-shelf OCR vs. fine-tuned TrOCR) and for linking each text to a component.
- Listing the alternatives I did not run, and why (the "Alternatives considered" section of the report).

**2. Exploration: understanding the data and things to have in consideration**

- What to check before training: image formats, COCO box format `[x, y, w, h]`, boxes outside the image or degenerate, class imbalance.
- How large the symbols become after resizing. This box-size check is what justifies `imgsz=1024` instead of 640.
- How to split without leakage (by image, stratified on each image's rarest class) and how to confirm that every class appears in every split.

**3. Understanding concepts**

- Evaluation: IoU, mAP50 vs. mAP50-95, why mAP is computed at `conf=0.001`, precision/recall at the best-F1 point, and how to choose an operating threshold on validation.
- Reading the Ultralytics confusion matrix (`matrix[predicted, true]`, background row and column).
- Why the test set must be used only once, and how to enforce it (`RUN_TEST`, `TEST_DONE.txt`).
- Training details: transfer learning, early stopping, batch size vs. GPU memory, and which augmentations make sense for drawings (no horizontal flips).
- For the bonus: how TrOCR is trained and why the built-in loss of transformers 4.57 shifted the labels twice.

**4. Explanation: making the notebooks readable**

- Markdown cells that explain what each step does, why, and how to verify it, for a reader without prior knowledge.
- Interpreting my own results, for example where the 10.9 % zero-shot score appears, why fine-tuning is needed, and why the bonus needs both the component detector and the text detector.

**5. Code: writing and fixing it**

- Drafting parts of the notebooks, which I then ran on Kaggle, checked and adjusted:
  - the COCO → YOLO conversion, with clipping and normalised boxes;
  - the stratified split by image;
  - the evaluation function (metrics, threshold choice, confusion matrix);
  - the error-analysis cells and the overlay of predictions on the ground truth.
- For the bonus: the TrOCR fine-tuning loop with an explicit loss, the unit-aware matching, the end-to-end evaluation and the example figure.
- Small utilities and fixes:
  - saving and zipping the outputs;
- Debugging errors and version issues, for example pinning `transformers` 4.57.6 because version 5 could not load the TrOCR tokenizer.

**6. Review: finding my own mistakes**

- Static reviews of both notebooks, without running them: bugs, protocol issues, and inconsistencies between the markdown, the code and the README (prompts 1-3 below).
- A critical look at whether the bonus pipeline was over-engineered, and what could be simpler.

**7. Delivery**

- Workflow between Kaggle and GitHub, and how someone else can reproduce the results.
- Structure of the repository, README and report. Every number in them comes from my Kaggle outputs (`results_summary.json`, `values_summary.json`).

## Main prompts and what I kept

| Step                   | What I asked                                                                                                    | What I kept                                                                                                          |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Task 1 notebook review | A static review of `circuit-detection.ipynb`: bugs, evaluation protocol, markdown explanations (prompt 1 below) | The bug fixes and evaluation changes listed in response 1                                                            |
| Final notebook review  | Review the final Task 1 notebook without running it, and improve the markdown                                   | Wording of the markdown cells, and a fix so the output cell works with `RUN_TEST = False`                            |
| Bonus notebook review  | A static review of `value-reading.ipynb`                                                                        | Robustness fixes: unique file paths, dropping images without a split, creating the output folder, freeing GPU memory |
| Packaging              | Structure of the GitHub repository, README and report wording                                                   | Repository layout, README, report text (numbers taken from my own runs)                                              |

---

## Prompt 1: Task 1 notebook review (translated and shortened from the Spanish original)

> **Role.** Act as a senior computer-vision engineer specialised in Ultralytics/YOLO and in detector evaluation. Review `circuit_detection.ipynb`, fix what is wrong and leave it in a state that can be defended in a technical interview. I don't want a generic list of advice. I want: (1) concrete bugs, with cell and line; (2) justified fixes; (3) markdown that explains what each cell does, why, and the concept behind it, without assuming prior knowledge.
>
> **Project context.**
>
> - Task: detect and classify every component of a hand-drawn circuit diagram (17 classes, box + label).
> - Dataset: Digitize-HCD, 1,277 images scanned at 600 dpi, COCO annotations in `component_annotations.json`.
> - Environment: Kaggle, GPU T4, 12 h per session, Internet on. No paid solutions.
> - Current notebook: YOLO11m pretrained on COCO, `imgsz=1024`, 100 epochs, `patience=20`, `batch=4`, 80/10/10 split by image, seed 42.
> - The test set must be used once, after all decisions are fixed.
>
> **Technical checklist (summary).**
>
> - *Data and paths:* image extension (`glob("*.jpg")` on PNG is a bug), dataset integrity, COCO `[x, y, w, h]` boxes, out-of-bounds and degenerate boxes, COCO→YOLO conversion (class offset, normalisation, clipping, `data.yaml`).
> - *Split:* by image, never by box; is "rarest class per image" stratification correct; save the exact split; stale files in the output folder could cause silent leakage on re-runs.
> - *Training:* `batch=4` at 1024 px on a 16 GB T4 (out-of-memory risk); time per epoch; reproducibility (seed, pinned Ultralytics version, `requirements.txt`); augmentation (`fliplr=0`, `degrees`, mosaic); `exist_ok=True` could reuse an old `best.pt`; justify `imgsz=1024` with box-size percentiles.
> - *Evaluation:* `conf=0.001` is right for AP, but the confusion matrix needs a real threshold; `iou=0.6` in `model.val` is an NMS setting, not the mAP threshold; `m.box.mp/mr` are at the best-F1 point; add `n_gt` per class; check the confusion-matrix orientation; guard so the test set runs once; save metrics and predictions.
> - *Error analysis:* overlay ground truth and predictions; analyse small components, rare classes and look-alike pairs (BJT NPN/PNP, Diode/Zener, MOSFET N/P).
> - *Model justification:* the reference thesis used YOLOv8, which is not direct evidence for YOLO11m; either compare on validation or state that there is no own benchmark.
> - *Deliverables:* README, `LLM_USAGE.md`, 1-2 page report, `requirements.txt`, inference on a new image, dataset citation and licences.
>
> **Rules.** Static review only: do not run the notebook and do not invent results. Do not use the test set for tuning. Keep the overall strategy; propose big changes as optional. Classify findings as P0 (breaks execution or invalidates results), P1 (protocol or reproducibility) or P2 (clarity).
>
> **Deliverables.** Executive summary, table of findings by severity, edited notebook with a backup, change log, static checks for me to run, open points.

## Response 1 (summary of the changes made)

> Same approach: YOLO11m, 1024 px, 100 epochs / patience 20, 80/10/10 split by image, seed 42.
>
> **Bugs fixed**
>
> - Images were searched only as `*.jpg`; they are now listed with any extension.
> - The dataset path was hard-coded for one Kaggle account; it is now found automatically.
> - The YOLO folder was never deleted, so a re-run could leave old images in the wrong split. It is now rebuilt from scratch.
> - With `exist_ok=True`, a failed run could silently evaluate an older `best.pt`. Each run now gets its own folder.
> - Boxes are clipped to the image before removing tiny ones.
> - COCO category ids are mapped to 0..16 explicitly instead of `id - 1`.
>
> **Evaluation**
>
> - The confidence threshold is chosen on validation (best F1) and reused for test and predictions.
> - The confusion matrix is computed at that threshold instead of 0.001.
> - P and R are labelled as "best F1"; `iou=0.6` was removed.
> - The per-class table shows `n_gt`.
> - The test set only runs with `RUN_TEST = True` and is recorded in `TEST_DONE.txt`.
> - Predictions are shown with the ground truth on top.
>
> **Other**
>
> - Ultralytics pinned to 8.4.182; `requirements.txt` saved.
> - `batch=4` instead of 8, to avoid running out of memory on the T4.
> - The box-size check justifies `imgsz=1024`.
> - `split.csv` and per-class metrics are saved.

---

## Prompt 2: final review of the Task 1 notebook

> Focus on this file. I want you to review it, not run it. Help me make the markdown annotations perfect, and fix anything I missed.

**Response 2 (summary).** Static review, no execution. Fixed: the output cell copied test-only files unconditionally (crash with `RUN_TEST = False`) and built the zip twice; `split.csv` gained a `file_name` column; the count of images without valid boxes is printed. The markdown was rewritten: a single T4 (the code uses `device=0`), the effect of batch 4 on BatchNorm, why small strata are merged at 10 images, the leakage caveat, and how P/R relate to the chosen threshold. I kept my own version of the final "test on my own image" cell.

## Prompt 3: review of the bonus notebook

> Review `value-reading.ipynb` statically (do not run it): correctness bugs, consistency between markdown and code, consistency with the README, and markdown quality. Only fix real problems.

**Response 3 (summary).** The pipeline logic was judged correct (no leakage between splits, test evaluated once). Robustness fixes:

- `glob(...)[0]` could silently pick the wrong `best.pt` or `split.csv`; a helper now requires exactly one match.
- Text boxes from images missing in `split.csv` would get no split and crash the text-detector cell; they are now dropped and counted (0 in my run).
- The output folder is created at the start.
- GPU memory from the TrOCR optimiser is freed before training the text detector.

It also pointed out that "attached to a compatible class" does not prove the right component was chosen; the README and report now say so.

## Prompt 4: packaging

> Help me upload the project to GitHub with the results zip, and edit the README and the other files in the repository.

**Response 4 (summary).** A repository layout (`task1_detection/`, `task2_value_reading/`, results and figures, weights in a Release), this README, and the report text. All numbers come from my Kaggle runs (`results_summary.json`, `values_summary.json`).

## Prompt 5: understanding and checking my own choices

Short questions I asked while finishing the task, with the gist of each answer:

| Question                                                                   | Answer (summary)                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is there over-engineering in the bonus task? Could it be done more simply? | No: each part is backed by a number (text detector mAP50 0.995; fine-tuning takes the reader from 10.9 % to ~91 % exact match; the unit rule changes 5.1 % of the choices). Simpler options exist (text as an 18th YOLO class, distance-only matching, off-the-shelf OCR), each with a cost. |
| Why are we using the other model too? To identify the components again?    | Yes. The Task 1 detector finds the components, the text detector finds where the text is and TrOCR reads it. The matching joins the two outputs.                                                                                                                                             |
| Where in the notebook is the 10 % that shows we need to fine-tune?         | Section 4 prints the zero-shot score on validation (exact match 0.109); section 5 prints the fine-tuned score (0.916).                                                                                                                                                                       |
| Would it be better to have just one notebook?                              | No: the bonus is optional, so it can be judged separately, and a combined notebook would take ~3 h and fail as a whole.                                                                                                                                                                      |                                                                                                                                                                                                                                                                          |
