# Evaluation Module: What Each File Does and Input/Output

The **evaluation** folder holds code to score your tracking results against ground truth (e.g. MOTA, IDF1, RMSE, **distance from GT**, **ID switches**). Below is where to find each file and what it does, plus the exact input and output formats.

**Ground truth is in the camera/image frame.** When using the CARLA pipeline, 3D pedestrian positions are projected into 2D using the camera intrinsics, so the GT file contains bboxes in the same pixel coordinates as the video and the tracker output. The evaluation script then compares tracker predictions to this image-frame GT (distance metrics in pixels, plus IDSW).

---

## Where the files are

From the **repository root** (the folder that contains `perception/`, `carla_integration/`, etc.), the evaluation files live here:

```
evaluation/
├── README.md          ← this file (overview + input/output)
├── __init__.py        ← package init, re-exports public functions
├── metrics.py         ← core metric math (MOTA, MOTP, IDF1, RMSE, etc.)
├── io.py              ← load/save and convert data (MOT files, frame lists)
└── run_metrics.py     ← CLI script to compute metrics from two files
```

If you don’t see `evaluation/`, check that you’re at the repo root and that the folder wasn’t excluded (e.g. by `.gitignore`). It should sit next to `perception/` and `carla_integration/`.

---

## What each file does

### 1. `evaluation/metrics.py`

**Role:** Implements the metrics and matching logic.

- **MOT metrics (CLEAR-style):** MOTA, MOTP, IDF1, IDSW, FP, FN. Uses IoU-based matching (Hungarian) per frame and counts identity switches.
- **Trajectory metrics:** RMSE, ADE, FDE (center distance between matched track and GT over time).

**Input (for the main function `compute_metrics`):**

- **`gt_frames`**  
  A list of **frames**. Each frame is a list of **ground-truth objects**:  
  `(gt_id, bbox_xyxy)`  
  where `bbox_xyxy = (x1, y1, x2, y2)` in image coordinates.
- **`pred_frames`**  
  Same structure for **predictions**: each frame is a list of  
  `(track_id, bbox_xyxy)`.

So:

- **Input type:** Two Python lists of lists of `(id, (x1,y1,x2,y2))`.
- **Optional:** `min_iou` (default 0.5) for considering a detection matched to GT.

**Output:**

- A single **dict** with keys:  
  `MOTA`, `MOTP`, `IDF1`, `IDSW`, `FP`, `FN`, `num_matches`, `total_gt`,  
  `RMSE`, `ADE`, `FDE`, `num_trajectory_points`.  
  Values are floats (or ints for counts). Use `format_metrics(metrics)` to get a readable string.

---

### 2. `evaluation/io.py`

**Role:** Get data *into* the format `metrics.py` expects, and write results out.

- **`load_mot_file(path)`**  
  Reads a **MOTChallenge-style** text file (one line per detection):  
  `frame,id,left,top,width,height`  
  and returns a **list of frames** in the same format as `gt_frames` / `pred_frames` above (so you can pass it straight into `compute_metrics`).

  - **Input:** Path to a `.txt` file (e.g. `gt/gt.txt` or your tracker output).
  - **Output:** `list of list of (id, (x1,y1,x2,y2))` (frames 0-based).

- **`frames_from_boxes_and_ids(gt_boxes_per_frame, pred_boxes_ids_per_frame)`**  
  Builds `gt_frames` and `pred_frames` when you have **per-frame lists of boxes** (e.g. from CARLA-MOT or your own loop).

  - **Input:**  
    - `gt_boxes_per_frame`: list of lists. Each inner list = one frame; each element = bbox `(x1,y1,x2,y2)` or `(x1,y1,x2,y2, id)`.  
    - `pred_boxes_ids_per_frame`: list of lists. Each inner list = one frame; each element = `(x1,y1,x2,y2, track_id)` or `(bbox, track_id)`.  
  - **Output:** `(gt_frames, pred_frames)` in the format expected by `compute_metrics`.

- **`save_metrics_report(metrics, path)`**  
  Writes the metrics dict to a text file (same content as `format_metrics`).

  - **Input:** The dict returned by `compute_metrics` and a file path.  
  - **Output:** A file on disk; no return value.

---

### 3. `evaluation/run_metrics.py`

**Role:** Command-line tool to compute metrics from two MOT-style files.

**Input (command line):**

- **`--gt`** (required): Path to ground truth file (MOT format: `frame,id,left,top,width,height`).
- **`--pred`** (required): Path to tracker output file (same format).
- **`--out`** (optional): If set, the metrics report is written to this file.
- **`--min-iou`** (optional, default 0.5): Minimum IoU to count a match.

**Output:**

- **To terminal:** Printed metrics (MOTA, MOTP, IDF1, IDSW, FP, FN, RMSE, ADE, FDE).
- **To file:** If `--out report.txt` is given, the same report is saved to `report.txt`.

**Example (run from repo root):**

```bash
python evaluation/run_metrics.py --gt path/to/gt/gt.txt --pred path/to/results.txt --out report.txt
```

---

### 4. `evaluation/__init__.py`

**Role:** Defines the package and what gets imported with `from evaluation import ...`.

- Re-exports: `compute_metrics`, `compute_mot_metrics`, `compute_trajectory_metrics`, `format_metrics`, `load_mot_file`, `save_metrics_report`, `frames_from_boxes_and_ids`.  
- No input/output; it only affects imports.

---

## Input/Output summary

| What you have | What you use | What you get |
|---------------|--------------|--------------|
| Two MOT-style files (GT + predictions) | `load_mot_file` → `compute_metrics` or `run_metrics.py` | Dict (or report file) with MOTA, MOTP, IDF1, IDSW, FP, FN, RMSE, ADE, FDE |
| Per-frame lists of boxes (e.g. from CARLA) | `frames_from_boxes_and_ids` → `compute_metrics` | Same metrics dict |
| A metrics dict | `format_metrics` or `save_metrics_report` | Human-readable string or report file |

**Data format reminder:** Each frame is a list of `(id, (x1, y1, x2, y2))` with `x1,y1` = top-left and `x2,y2` = bottom-right of the bounding box.
