"""
Central configuration for the experiment pipeline.
All tuneable knobs live here so runs are reproducible.
"""
import math

# ── Method definitions ────────────────────────────────────────────────────
METHODS = [
    {"name": "imm_l0",  "filter": "imm", "lambda_": 0.0, "label": "Behavioral IMM (λ=0)"},
    {"name": "imm_l05", "filter": "imm", "lambda_": 0.5, "label": "Behavioral IMM (λ=0.5)"},
    {"name": "kf_l0",   "filter": "kf",  "lambda_": 0.0, "label": "Vanilla KF (λ=0)"},
    {"name": "kf_l05",  "filter": "kf",  "lambda_": 0.5, "label": "Vanilla KF (λ=0.5)"},
    {"name": "kf_l1",   "filter": "kf",  "lambda_": 1.0, "label": "Appearance Only (λ=1)"},
]

# ── Sweep parameters ─────────────────────────────────────────────────────
PED_COUNTS = [1, 2, 3, 4, 5]
N_TRIALS = 1            # increase for statistical robustness

# ── CARLA scenario settings ──────────────────────────────────────────────
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
MAP_NAME = "Town10HD"
SPAWN_INDEX = 0         # spawn_points[0]

SIM_DURATION = 5.0     # seconds
FPS = 20
WIDTH = 800
HEIGHT = 600
FOV = 90

VEHICLE_BP = "vehicle.tesla.model3"

# Pedestrian spawn distribution (relative to ego, in CARLA frame)
SPAWN_MU = (6.0, 0.0, 0.0)      # (forward-x, right-y, up-z)
SPAWN_SIGMA = (1.0, 3.0, 0.0)    # std-dev per axis; z=0 keeps peds on ground
PED_SPEED_RANGE = (1.0, 1.5)     # m/s uniform range

# Social force parameters (pedestrian repulsion)
SOCIAL_FORCE_RADIUS = 2.0        # metres — interaction distance
SOCIAL_FORCE_STRENGTH = 3.5      # repulsion magnitude
SOCIAL_FORCE_FALLOFF = 0.5       # exponential decay rate (1/metres)

# ── Camera intrinsics (derived from WIDTH/HEIGHT/FOV) ────────────────────
FX = WIDTH / (2.0 * math.tan(math.radians(FOV / 2.0)))   # 400.0 for 800/90°
FY = FX
CX = WIDTH / 2.0    # 400.0
CY = HEIGHT / 2.0   # 300.0

# Camera mounting on vehicle (matches spawn_pred_crossing_with_Depth.py)
CAM_X = 1.5   # metres forward
CAM_Z = 2.0   # metres up
CAM_PITCH = 0.0

MAX_DEPTH_M = 50.0

# ── Tracking settings ────────────────────────────────────────────────────
YOLO_WEIGHTS = "yolov9-c.pt"
COSINE_THRESHOLD = 0.6
NN_BUDGET = 100
CONF_THRESH = 0.8
IOU_THRESH = 0.15

# ── Evaluation settings ──────────────────────────────────────────────────
COV_ELLIPSE_INTERVAL_S = 0.5     # draw covariance ellipse every 0.5 s
COV_ELLIPSE_INTERVAL_FRAMES = int(COV_ELLIPSE_INTERVAL_S * FPS)  # 10 frames
