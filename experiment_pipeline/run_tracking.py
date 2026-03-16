#!/usr/bin/env python
"""
Unified tracking script for the experiment pipeline.

Runs YOLOv9 detection + OSNet ReID + DeepSORT tracking with a configurable
filter (behavioral IMM or vanilla KF) and lambda (appearance vs motion weight).

Outputs:
  - tracks_world.csv : per-frame world-frame position + covariance per track
  - tracks_mot.txt   : standard MOT-format 2D bounding boxes

Usage:
  python run_tracking.py --source video.mp4 --depth_dir depth_frames/ \\
      --filter imm --lambda_ 0.5 --output_dir results/n3/imm_l05
"""
from __future__ import print_function

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchreid

# ── Path setup ────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_PERCEPTION = _REPO_ROOT / "perception"
_YOLO_ROOT = _PERCEPTION / "yolov9"
_DEEP_SORT_ROOT = _PERCEPTION / "deep_sort"

for p in [_DEEP_SORT_ROOT, _YOLO_ROOT, _PERCEPTION, str(_REPO_ROOT)]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as IMMTracker
from deep_sort.tracker_vanilla_kf import Tracker as KFTracker

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

from experiment_pipeline.config import (
    FX, FY, CX, CY, MAX_DEPTH_M,
    COSINE_THRESHOLD, NN_BUDGET, CONF_THRESH, IOU_THRESH, YOLO_WEIGHTS,
)


# ── Depth loading ────────────────────────────────────────────────────

def load_depth_image(depth_path):
    """Load a 16-bit PNG depth image (uint16_mm encoding) -> float32 metres."""
    img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    depth_m = img.astype(np.float32) / 65535.0 * MAX_DEPTH_M
    return depth_m


def unproject_pixel(u, v, depth_m, fx, fy, cx, cy):
    """Back-project pixel (u,v) with depth to 3D camera-frame coords [pX, pZ, pY]."""
    pX = (u - cx) * depth_m / fx
    pY = (v - cy) * depth_m / fy
    pZ = depth_m
    return np.array([pX, pZ, pY], dtype=np.float64)


# ── World-frame state extraction ─────────────────────────────────────

def extract_imm_world_state(track):
    """Extract fused world position and 3x3 position covariance from IMM packed state.

    Packed layout (24-D):
      mean[ 0: 7] = CV  [pX, pZ, v, phi, omega, pY, vY]
      mean[ 7:14] = CT  [pX, pZ, v, phi, omega, pY, vY]
      mean[14:21] = CA  [pX, pZ, v, phi, omega, pY, vY]
      mean[21:24] = mode probabilities [mu_cv, mu_ct, mu_ca]
    Position indices within each 7D block: pX=0, pZ=1, pY=5
    """
    m = track.mean
    mu = m[21:24]
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    if s > 1e-12:
        mu = mu / s
    else:
        mu = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    pos_idx = [0, 1, 5]  # pX, pZ, pY within each 7D block
    offsets = [0, 7, 14]

    # Fused position
    world_pos = np.zeros(3)
    positions = []
    for j, off in enumerate(offsets):
        p = np.array([m[off + k] for k in pos_idx])
        positions.append(p)
        world_pos += mu[j] * p

    # Fused position covariance (IMM mixing formula)
    C = track.covariance
    P_fused = np.zeros((3, 3))
    for j, off in enumerate(offsets):
        idx = [off + k for k in pos_idx]
        P_j = C[np.ix_(idx, idx)]
        d = positions[j] - world_pos
        P_fused += mu[j] * (P_j + np.outer(d, d))

    return world_pos, P_fused  # [pX, pZ, pY], 3x3


def extract_kf_world_state(track, depth_img, fx, fy, cx, cy):
    """Extract world position and approximate 3x3 covariance from vanilla KF track + depth.

    KF state: [x, y, a, h, vx, vy, va, vh] in image frame.
    """
    u = track.mean[0]
    v = track.mean[1]

    dh, dw = depth_img.shape[:2]
    u_d = int(np.clip(u, 0, dw - 1))
    v_d = int(np.clip(v, 0, dh - 1))
    d = float(depth_img[v_d, u_d])

    if d < 0.1:
        return None, None

    world_pos = unproject_pixel(u, v, d, fx, fy, cx, cy)

    # Jacobian of unprojection w.r.t. (u, v)
    J = np.array([
        [d / fx, 0.0],
        [0.0,    0.0],
        [0.0,    d / fy],
    ])
    P_img = track.covariance[:2, :2]
    P_world = J @ P_img @ J.T
    # Add small depth uncertainty along Z axis
    P_world[1, 1] += (0.05 * d) ** 2

    return world_pos, P_world  # [pX, pZ, pY], 3x3


# ── Main tracking function ───────────────────────────────────────────

def run_tracking(source, depth_dir, filter_type, lambda_val, output_dir,
                 weights=None, device_str="", fx=FX, fy=FY, cx=CX, cy=CY,
                 save_video=False, conf_thresh=CONF_THRESH):
    """Run detection + tracking pipeline and save world-frame results."""

    os.makedirs(output_dir, exist_ok=True)
    tracks_world_path = os.path.join(output_dir, "tracks_world.csv")
    tracks_mot_path = os.path.join(output_dir, "tracks_mot.txt")
    video_out_path = os.path.join(output_dir, "tracking.mp4") if save_video else None

    if weights is None:
        weights = str(_PERCEPTION / "yolov9" / "weights" / YOLO_WEIGHTS)

    # Device
    if device_str and "cuda" in device_str and torch.cuda.is_available():
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[tracking] device={device}  filter={filter_type}  lambda={lambda_val}")

    # YOLO
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)

    # ReID
    reid_model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True
    )
    reid_model.eval().to(device)
    reid_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Depth files
    depth_dir_path = Path(depth_dir)
    depth_files = sorted(depth_dir_path.glob("*.png"))
    print(f"[tracking] {len(depth_files)} depth frames in {depth_dir}")

    # Tracker
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric="cosine", matching_threshold=COSINE_THRESHOLD, budget=NN_BUDGET
    )
    use_depth = (filter_type == "imm")
    if filter_type == "imm":
        tracker = IMMTracker(metric, fallback_mode="3d", fallback_3d_threshold=0.6, lambda_=lambda_val)
    else:
        tracker = KFTracker(metric, lambda_=lambda_val)

    # Data loader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    model.warmup(imgsz=(1, 3, *imgsz))

    depth_frame_idx = 0
    frame_no = 0

    world_file = open(tracks_world_path, "w", newline="")
    world_writer = csv.writer(world_file)
    world_writer.writerow([
        "frame", "track_id",
        "world_pX", "world_pZ", "world_pY",
        "cov_00", "cov_01", "cov_02",
        "cov_10", "cov_11", "cov_12",
        "cov_20", "cov_21", "cov_22",
    ])

    mot_file = open(tracks_mot_path, "w")

    # Video writer (lazy init on first frame)
    vid_writer = None
    _TRACK_COLORS = [
        (230, 100, 50), (50, 180, 50), (50, 50, 230), (200, 200, 50),
        (200, 50, 200), (50, 200, 200), (255, 150, 0), (0, 150, 255),
        (150, 0, 255), (100, 255, 100),
    ]

    try:
        for path, im, im0s, vid_cap, s in dataset:
            frame_no += 1

            im_tensor = torch.from_numpy(im).to(device).float() / 255.0
            if len(im_tensor.shape) == 3:
                im_tensor = im_tensor[None]

            pred = model(im_tensor, augment=False)[0][1]
            pred = non_max_suppression(pred, conf_thresh, IOU_THRESH, max_det=1000)

            # Load depth
            depth_img = None
            if depth_frame_idx < len(depth_files):
                depth_img = load_depth_image(depth_files[depth_frame_idx])
            depth_frame_idx += 1

            im0 = im0s.copy()
            vis_frame = im0.copy() if save_video else None

            # Lazy-init video writer from first frame dimensions
            if save_video and vid_writer is None:
                h_frame, w_frame = im0.shape[:2]
                src_fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(video_out_path, fourcc, src_fps, (w_frame, h_frame))

            for det in pred:
                if len(det) == 0:
                    tracker.predict()
                    tracker.update([])
                    continue

                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                detections = []
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) != 0:  # COCO class 0 = person; skip everything else
                        continue
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cx_det = int((x1 + x2) / 2)
                    cy_det = int((y1 + y2) / 2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    x_tl, y_tl = int(cx_det - w / 2), int(cy_det - h / 2)
                    w, h = int(w), int(h)

                    crop = im0[y_tl:y_tl + h, x_tl:x_tl + w]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = reid_transform(crop_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = reid_model(input_tensor)
                    feature = feat.cpu().numpy().flatten()
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        feature = feature / norm

                    tlwh = [x_tl, y_tl, w, h]

                    world_pos = None
                    if use_depth and depth_img is not None:
                        dh, dw = depth_img.shape[:2]
                        u_d = int(np.clip(cx_det, 0, dw - 1))
                        v_d = int(np.clip(cy_det, 0, dh - 1))
                        d = float(depth_img[v_d, u_d])
                        if d > 0.1:
                            world_pos = unproject_pixel(cx_det, cy_det, d, fx, fy, cx, cy)

                    detections.append(Detection(tlwh, float(conf), feature, world_pos=world_pos))

                tracker.predict()
                tracker.update(detections)

                # Extract outputs
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    tid = track.track_id
                    x1t, y1t, wt, ht = track.to_tlwh()

                    # MOT line
                    mot_file.write(f"{frame_no},{tid},{x1t:.2f},{y1t:.2f},{wt:.2f},{ht:.2f},1\n")

                    # Draw on video frame
                    if vis_frame is not None:
                        color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
                        bx1, by1 = int(x1t), int(y1t)
                        bx2, by2 = int(x1t + wt), int(y1t + ht)
                        cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), color, 2)
                        label = f"ID {tid}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_frame, (bx1, by1 - th - 4), (bx1 + tw, by1), color, -1)
                        cv2.putText(vis_frame, label, (bx1, by1 - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                    # World-frame extraction
                    w_pos, w_cov = None, None
                    if filter_type == "imm":
                        w_pos, w_cov = extract_imm_world_state(track)
                    elif depth_img is not None:
                        w_pos, w_cov = extract_kf_world_state(
                            track, depth_img, fx, fy, cx, cy
                        )

                    if w_pos is not None and w_cov is not None:
                        world_writer.writerow([
                            frame_no, tid,
                            f"{w_pos[0]:.6f}", f"{w_pos[1]:.6f}", f"{w_pos[2]:.6f}",
                            f"{w_cov[0,0]:.6f}", f"{w_cov[0,1]:.6f}", f"{w_cov[0,2]:.6f}",
                            f"{w_cov[1,0]:.6f}", f"{w_cov[1,1]:.6f}", f"{w_cov[1,2]:.6f}",
                            f"{w_cov[2,0]:.6f}", f"{w_cov[2,1]:.6f}", f"{w_cov[2,2]:.6f}",
                        ])

            if vid_writer is not None and vis_frame is not None:
                vid_writer.write(vis_frame)

            if frame_no % 50 == 0:
                print(f"  [tracking] frame {frame_no}")

    finally:
        world_file.close()
        mot_file.close()
        if vid_writer is not None:
            vid_writer.release()
            print(f"  tracking_vid -> {os.path.abspath(video_out_path)}")
        print(f"[tracking] Done. {frame_no} frames processed.")
        print(f"  tracks_world -> {os.path.abspath(tracks_world_path)}")
        print(f"  tracks_mot   -> {os.path.abspath(tracks_mot_path)}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment tracking pipeline")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--depth_dir", required=True, help="Path to depth frames dir")
    parser.add_argument("--filter", choices=["imm", "kf"], default="imm")
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH,
                        help="YOLO confidence threshold (default from config)")
    parser.add_argument("--save_video", action="store_true",
                        help="Save a video with bounding boxes and track IDs")
    parser.add_argument("--fx", type=float, default=FX)
    parser.add_argument("--fy", type=float, default=FY)
    parser.add_argument("--cx", type=float, default=CX)
    parser.add_argument("--cy", type=float, default=CY)
    args = parser.parse_args()

    run_tracking(
        source=args.source,
        depth_dir=args.depth_dir,
        filter_type=args.filter,
        lambda_val=args.lambda_,
        output_dir=args.output_dir,
        weights=args.weights,
        device_str=args.device,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        save_video=args.save_video,
        conf_thresh=args.conf_thresh,
    )


if __name__ == "__main__":
    main()
