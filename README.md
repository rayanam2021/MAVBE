# MAVBE
Multi-Agent Vehicle and Behavior Estimator (MAVBE)

---

## Which files to run

### Detection + tracking (video / webcam / images)

| What you want | File to run | Where to run from | Notes |
|---------------|-------------|-------------------|--------|
| **YOLO + Deep SORT** on a video, folder, or webcam | `detect_dual_tracking.py` | Repo root `MAVBE/` | Uses `deep_sort_pytorch` and YOLO (`models/`, `utils/`). Ensure `perception/yolov9` (or your YOLO root) is on `PYTHONPATH`, or run from `perception/yolov9` with `python ../../detect_dual_tracking.py --source <video_or_folder>`. |

**Example (from repo root):**
```bash
# From MAVBE/ (adjust path to yolo weights and source as needed)
python detect_dual_tracking.py --source path/to/video.mp4 --weights perception/yolov9/yolov9-c.pt
```

---

### Deep SORT with Behavioral EKF (MOTChallenge-style sequences)

The in-repo tracker uses a **Behavioral EKF** (CT + social force) and runs on precomputed detections in MOTChallenge layout.

| What you want | File to run | Where to run from | Notes |
|---------------|-------------|-------------------|--------|
| **Run tracker** on one sequence (with display) | `perception/deep_sort/deep_sort_app.py` | `perception/deep_sort/` | Needs a sequence dir (e.g. `img1/`, `seqinfo.ini`) and a detection `.npy` file. |
| **Evaluate** on a full MOT dataset | `perception/deep_sort/evaluate_motchallenge.py` | `perception/deep_sort/` | Runs the tracker on every sequence under `--mot_dir` and writes results to `--output_dir`. |
| **Generate detections** (MOT format + features) | `perception/deep_sort/tools/generate_detections.py` | `perception/deep_sort/` | Produces `.npy` detections for use with `deep_sort_app.py`. Needs a frozen model (e.g. `mars-small128.pb`) and `--mot_dir`. |
| **Visualize** tracking results | `perception/deep_sort/show_results.py` | `perception/deep_sort/` | Sequence dir + result file in MOT format. |
| **Generate videos** from results | `perception/deep_sort/generate_videos.py` | `perception/deep_sort/` | Batch video generation from MOT-style outputs. |

**Examples (from `perception/deep_sort/`):**
```bash
# Single sequence (behavioral EKF tracker)
python deep_sort_app.py --sequence_dir=./MOT16/train/MOT16-02 --detection_file=./resources/detections/MOT16-02.npy --min_confidence=0.3 --display=True

# Evaluate all sequences in a MOT directory
python evaluate_motchallenge.py --mot_dir=./MOT16/train --detection_dir=./resources/detections --output_dir=./results
```

See `perception/deep_sort/README.md` for installation, detection generation, and MOT data layout.

---

### YOLOv9 (detection / training)

| What you want | File to run | Where to run from |
|---------------|-------------|--------------------|
| **Inference** (detect only) | `perception/yolov9/detect.py` | `perception/yolov9/` |
| **Train** detection model | `perception/yolov9/train.py` | `perception/yolov9/` |
| **Validate** | `perception/yolov9/val.py` | `perception/yolov9/` |

Run from `perception/yolov9/`; see `perception/yolov9/README.md` for data and options.

---

### CARLA integration

| What you want | File to run | Where to run from | Notes |
|---------------|-------------|-------------------|--------|
| **CARLA auto-control demo** (vehicle + sensors) | `carla_integration/trajectory_planning.py` | Repo root or `carla_integration/` | Requires CARLA server, Python API, and `agents.navigation` modules. Not yet wired to the MOT/Behavioral EKF pipeline. |

---

### Summary

- **Quick tracking on a video:** `detect_dual_tracking.py` (YOLO + Deep SORT; uses external `deep_sort_pytorch`).
- **Behavioral EKF tracker on MOT data:** run `perception/deep_sort/deep_sort_app.py` or `evaluate_motchallenge.py` (in-repo `perception/deep_sort` with Behavioral EKF).
- **Detection only / training:** use scripts under `perception/yolov9/`.
- **CARLA driving demo:** `carla_integration/trajectory_planning.py`.
