# MAVBE Cognitive Map

## 1. System Overview
* **Project Name:** MAVBE (Multi-Agent Vehicle and Behavior Estimator)
* **Core Purpose:** A multi-agent tracking and behavior estimation framework that integrates YOLOv9 for detection with a specialized Behavioral EKF tracker (Coordinated Turn + Social Force models) for improved motion prediction in autonomous driving scenarios.
* **Tech Stack:** Python 3.9+, PyTorch (YOLOv9), CARLA Simulator (0.9.16), OpenCV, NumPy, SciPy (Linear Algebra/Filters).

## 2. Architecture & Data Flow
* **System Architecture:** The system follows a modular **Sense-Track-Predict** pipeline. It bridges high-fidelity simulation (CARLA) with computer vision perception (YOLOv9) and advanced state estimation (Behavioral EKF).
* **Data Paths:**
    * **Sim-to-Perception:** `carla_integration/spawn_pedestrian_video.py` generates `.mp4` data from CARLA -> Ingested by `perception/detect_dual_tracking.py`.
    * **Perception Pipeline:** Raw Video -> `YOLOv9` (Detection) -> `DeepSORT Tracker` (Association) -> `BehavioralEKF` (State Prediction & Social Force calculation) -> Trajectory outputs.
    * **Offline Evaluation:** MOTChallenge detections (`.npy`) -> `perception/deep_sort/deep_sort_app.py` -> `BehavioralEKF` tracking -> Results evaluation via `evaluate_motchallenge.py`.
* **APIs & Interfaces:**
    * **CARLA Python API:** Interaction with the local CARLA 0.9.16 server for agent spawning and sensor telemetry.
    * **DeepSORT API:** Standardized detection and track objects for modularity between different filters.

## 3. Critical Components & Entry Points
### Perception & Tracking (The "Brain")
* **Entry Point:** `perception/detect_dual_tracking.py` - Primary script for live/video inference combining YOLOv9 and the in-repo tracker.
* **Tracker Core:** `perception/deep_sort/deep_sort/tracker.py` (`Tracker` class) - Manages track lifecycle, association, and invokes the prediction filter.
* **Behavioral EKF:** `perception/deep_sort/deep_sort/behavioral_ekf.py` (`BehavioralEKFFilter`) - **Critical Component.** Implements the Coordinated Turn (CT) model and Social Force repulsion logic for multi-agent prediction.

### Simulation & Control
* **Entry Point:** `carla_integration/spawn_pedestrian_video.py` - Scaffolds CARLA scenarios, spawns ego-vehicle + pedestrians, and records sensor data.
* **Entry Point:** `carla_integration/scenario_pedestrian_crossing.py` - Custom script for testing trackers with deterministic crossing scenarios.
* **Entry Point:** `carla_integration/trajectory_planning.py` - Implements vehicle navigation using CARLA's `BasicAgent` and `BehaviorAgent`.

### Utilities
* **Visualizers:** `perception/deep_sort/show_results.py` and `perception/deep_sort/application_util/visualization.py`.
* **Model Export:** `perception/yolov9/export.py` for converting weights.

## 4. Interdependencies & Configurations
* **Internal Dependencies:**
    * `perception/detect_dual_tracking.py` dynamically adds `perception/yolov9` and `perception/deep_sort` to `sys.path`.
    * The in-repo `deep_sort` module is specifically modified to use `BehavioralEKFFilter` instead of a standard Linear Kalman Filter.
* **External Dependencies:**
    * **CARLA:** Requires a running CARLA server (default `localhost:2000`). Local installation at `~/autonomy_projects/CARLA_0.9.16`.
    * **YOLOv9 Weights:** Requires `yolov9-c.pt` or similar in `perception/yolov9/weights/`.
* **Configuration:**
    * `configs/deep_sort.yaml`: Parameters for the tracker (max age, confidence thresholds).
    * `configs/deep_sort.yaml` (Note: documentation suggests it's for `deep_sort_pytorch`, but used as a reference for in-repo params).

## 5. Agent Instructions & Gotchas
* **Python Path:** Always ensure `perception/yolov9` and `perception/deep_sort` are in your `PYTHONPATH` if running scripts outside their respective directories.
* **CARLA Version:** Strictly compatible with CARLA 0.9.16. Ensure the `carla` egg/module is correctly linked in `carla_integration/` scripts.
* **EKF Singularity:** The `BehavioralEKF` in `behavioral_ekf.py` contains logic to handle $\omega \approx 0$ (straight-line motion) to avoid division by zero in the CT model.
* **Social Force Scaling:** Repulsion forces are calculated in the image/pixel coordinate space in `detect_dual_tracking.py` but may need scaling for world-coordinate simulations in CARLA.
* **Coordinate Inversion:** Remember that in trajectory plotting, the Y-axis is typically inverted (`plt.gca().invert_yaxis()`) to match image space (origin top-left).
