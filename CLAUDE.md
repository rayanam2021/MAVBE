# MAVBE — Claude Working Notes

## What This Project Does
Pedestrian tracking from an AV POV:
**YOLOv9 detection** → **OSNet ReID features** → **DeepSORT tracker** → **Behavioral EKF motion model**

Goal: reduce ID switches and improve tracking consistency through better motion prediction / behavior modeling.

---

## Entry Points

| Script | Purpose | Run from |
|--------|---------|----------|
| `perception/detect_dual_tracking.py` | Main inference pipeline (video → tracked output) | `MAVBE/` |
| `perception/deep_sort/deep_sort_app.py` | MOTChallenge-format offline evaluation | `perception/deep_sort/` |
| `carla_integration/spawn_pedestrian_video.py` | Generate test video from CARLA | `carla_integration/` |
| `carla_integration/scenario_pedestrian_crossing.py` | Deterministic crossing scenarios | `carla_integration/` |

---

## Core File Map

```
perception/
  detect_dual_tracking.py          # Main entry: YOLO + ReID + Tracker
  deep_sort/
    deep_sort/
      tracker.py                   # Multi-target tracker (orchestration)
      track.py                     # Single track state + lifecycle
      behavioral_ekf.py            # BehavioralEKFFilter (current motion model)
      kalman_filter.py             # Standard CV Kalman (baseline / reference)
      nn_matching.py               # Cosine/Euclidean ReID gallery matching
      detection.py                 # Detection dataclass (tlwh + conf + feature)
      linear_assignment.py         # Hungarian algorithm + cascade matching
      iou_matching.py              # IOU fallback matching
    deep_sort_app.py               # MOTChallenge evaluation harness
```

---

## behavioral_ekf_2.py vs behavioral_ekf.py
`behavioral_ekf_2.py` is the **active** motion model (imported by tracker.py). It is the improved version:
- Complete Jacobian including `d/d_omega` terms and `F[3,4]=dt`
- Kalman gain via `scipy.linalg.solve` (numerically stable, no matrix inverse)
- Joseph form covariance update for numerical stability
- `_nearest_psd()` eigenvalue clamping ensures covariance stays PSD

`behavioral_ekf.py` is the older inactive version (kept as reference).

---

## State Representation
| Space | Dims | Variables |
|-------|------|-----------|
| DeepSORT 8D (external) | 8 | x, y, a, h, vx, vy, va, vh |
| Behavioral EKF 5D (internal) | 5 | px, py, v, phi, omega |

`BehavioralEKFFilter` bridges these two spaces on every predict/update call.

---

## Motion Model: Coordinated Turn (CT)
Located in `behavioral_ekf.py → BehavioralEKF`. Handles straight-line (omega≈0) and arc motion. Social force repulsion between pedestrians is **computed but currently not applied** (commented out in predict()).

---

## Matching Strategy
Two-stage cascade in `tracker.py._match()`:
1. **Confirmed tracks**: `0.5 * cosine_appearance + 0.5 * chi2-gated Mahalanobis`
2. **Unconfirmed + recent misses**: IOU fallback

---

## Future Motion Models (drop-in replacements for behavioral_ekf.py)
- **IMM** (Interacting Multiple Models): blend CA/CV/CT models dynamically
- **LSTM**: learned motion prior from trajectory data
- Convention: each replacement must implement `.initiate()`, `.predict()`, `.update()`, `.gating_distance()` with same signatures as `BehavioralEKFFilter`

---

## Known Issues
1. Social force not applied to state (commented out in behavioral_ekf_2.py:67-68)
3. Debug `print("Occlusions")` in track.py:135 — noisy
4. Covariance inflation `1.5^N` diverges fast for long occlusions
5. BehavioralEKF measurement noise R=(0.002)^2 is very small — high trust in raw detections

---

## Run Example
```bash
cd /home/adi_linux/autonomy_projects/MAVBE/perception
python detect_dual_tracking.py \
  --source /path/to/video.mp4 \
  --weights yolov9/weights/yolov9-c.pt \
  --save_plot_name my_run \
  --conf-thres 0.75 --iou-thres 0.65
```
Output: `runs/detect/trajectories_my_run.mp4` + `.png`
