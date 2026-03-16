#!/usr/bin/env python
"""
CARLA experiment scenario:
  - Ego vehicle at spawn_points[SPAWN_INDEX] in Town10HD, stationary
  - All traffic lights forced to red
  - RGB + depth cameras (800x600, FOV 90)
  - N pedestrians with normally-distributed spawn around (+10,0,0) relative to ego
  - Records: RGB video, depth frames (16-bit PNG), world-frame GT CSV

Usage:
  python carla_scenario.py --n_peds 3 --seed 42 --output_dir results/trial_0/n3
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import random
import sys
import time

import cv2
import numpy as np

try:
    import carla
except ImportError:
    print("CARLA module not found. Add CARLA PythonAPI to PYTHONPATH.")
    sys.exit(1)

# Append repo root so we can import config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiment_pipeline.config import (
    CARLA_HOST, CARLA_PORT, MAP_NAME, SPAWN_INDEX,
    SIM_DURATION, FPS, WIDTH, HEIGHT, FOV, VEHICLE_BP,
    SPAWN_MU, SPAWN_SIGMA, PED_SPEED_RANGE,
    CAM_X, CAM_Z, CAM_PITCH, MAX_DEPTH_M, FX, FY, CX, CY,
    SOCIAL_FORCE_RADIUS, SOCIAL_FORCE_STRENGTH, SOCIAL_FORCE_FALLOFF,
)


# ── Depth helpers (same as spawn_pred_crossing_with_Depth.py) ─────────

def decode_depth_metres(bgra_img):
    R = bgra_img[:, :, 2].astype(np.float32)
    G = bgra_img[:, :, 1].astype(np.float32)
    B = bgra_img[:, :, 0].astype(np.float32)
    normalised = (R + G * 256.0 + B * 65536.0) / (256.0 ** 3 - 1.0)
    return normalised * 1000.0


def depth_to_uint16(depth_m, max_depth=MAX_DEPTH_M):
    d = np.clip(depth_m, 0.0, max_depth)
    return (d / max_depth * 65535.0).astype(np.uint16)


# ── Projection helpers ────────────────────────────────────────────────

def build_projection_matrix(width, height, fov_deg):
    focal = width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    return np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)


def world_to_image(cam_transform, K, world_point, width, height):
    dx = world_point.x - cam_transform.location.x
    dy = world_point.y - cam_transform.location.y
    dz = world_point.z - cam_transform.location.z
    fwd = cam_transform.get_forward_vector()
    right = cam_transform.get_right_vector()
    up = cam_transform.get_up_vector()
    x_cam = dx * fwd.x + dy * fwd.y + dz * fwd.z
    y_cam = dx * right.x + dy * right.y + dz * right.z
    z_cam = dx * up.x + dy * up.y + dz * up.z
    if x_cam <= 0.1:
        return None
    u = K[0, 0] * (y_cam / x_cam) + K[0, 2]
    v = K[1, 1] * (-z_cam / x_cam) + K[1, 2]
    if 0 <= u < width and 0 <= v < height:
        return (u, v)
    return None


def world_to_camera_frame(cam_transform, world_loc):
    """Convert CARLA world Location to camera-frame [pX, pZ, pY] matching IMM convention.

    pX = camera-right  (metres)
    pZ = camera-forward / depth  (metres)
    pY = camera-down   (metres)
    """
    dx = world_loc.x - cam_transform.location.x
    dy = world_loc.y - cam_transform.location.y
    dz = world_loc.z - cam_transform.location.z
    fwd = cam_transform.get_forward_vector()
    right = cam_transform.get_right_vector()
    up = cam_transform.get_up_vector()
    pX = dx * right.x + dy * right.y + dz * right.z
    pZ = dx * fwd.x + dy * fwd.y + dz * fwd.z
    pY = -(dx * up.x + dy * up.y + dz * up.z)
    return pX, pZ, pY


def get_gt_row(walker, ped_id, cam_transform, K, width, height):
    """Return (cam_pX, cam_pZ, cam_pY, bb_left, bb_top, bb_w, bb_h) or None.

    Positions are in camera frame to match the tracker's output coordinate system.
    """
    loc = walker.get_location()
    pt = world_to_image(cam_transform, K, loc, width, height)
    if pt is None:
        return None
    u, v = pt
    ped_w, ped_h = 50, 150
    left = max(0, min(u - ped_w // 2, width - ped_w))
    top = max(0, min(v - ped_h, height - ped_h))
    pX, pZ, pY = world_to_camera_frame(cam_transform, loc)
    return (pX, pZ, pY, left, top, ped_w, ped_h)


# ── Main scenario ────────────────────────────────────────────────────

def run_scenario(n_peds, seed, output_dir,
                 host=CARLA_HOST, port=CARLA_PORT):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, "depth_frames")
    os.makedirs(depth_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "video.mp4")
    gt_path = os.path.join(output_dir, "gt_world.csv")

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()

    if not world.get_map().name.endswith(MAP_NAME):
        print(f"Loading map {MAP_NAME} ...")
        world = client.load_world(MAP_NAME)
        time.sleep(4)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    # Force all traffic lights to red
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        tl.set_state(carla.TrafficLightState.Red)
        tl.set_green_time(0.0)
        tl.set_red_time(9999.0)
        tl.freeze(True)

    bp_lib = world.get_blueprint_library()

    # ── Ego vehicle ──────────────────────────────────────────────
    vehicle_bp = bp_lib.filter(VEHICLE_BP)[0]
    vehicle_bp.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    spawn_tf = spawn_points[SPAWN_INDEX]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        raise RuntimeError(f"Could not spawn vehicle at spawn_points[{SPAWN_INDEX}]")
    vehicle.set_simulate_physics(False)
    world.tick()

    ego_tf = vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    print(f"Ego at {ego_loc}  yaw={ego_yaw:.1f}")

    fwd = carla.Vector3D(math.cos(math.radians(ego_yaw)),
                         math.sin(math.radians(ego_yaw)), 0)
    right = carla.Vector3D(-math.sin(math.radians(ego_yaw)),
                           math.cos(math.radians(ego_yaw)), 0)

    # ── Cameras ──────────────────────────────────────────────────
    cam_tf = carla.Transform(
        carla.Location(x=CAM_X, z=CAM_Z),
        carla.Rotation(pitch=CAM_PITCH),
    )

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(WIDTH))
    rgb_bp.set_attribute("image_size_y", str(HEIGHT))
    rgb_bp.set_attribute("fov", str(FOV))
    camera_rgb = world.spawn_actor(rgb_bp, cam_tf, attach_to=vehicle)

    rgb_holder = {"data": None}
    def on_rgb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb_holder["data"] = arr[:, :, :3].copy()  # BGRA -> BGR (cv2 native)
    camera_rgb.listen(on_rgb)

    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(WIDTH))
    depth_bp.set_attribute("image_size_y", str(HEIGHT))
    depth_bp.set_attribute("fov", str(FOV))
    camera_depth = world.spawn_actor(depth_bp, cam_tf, attach_to=vehicle)

    depth_holder = {"data": None}
    def on_depth(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        depth_holder["data"] = decode_depth_metres(arr)
    camera_depth.listen(on_depth)

    K = build_projection_matrix(WIDTH, HEIGHT, FOV)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))

    # ── Pedestrians ──────────────────────────────────────────────
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        walker_bps = bp_lib.filter("walker.*")

    walkers = []
    walker_ids = {}       # actor python-id -> ped_id (1-based)
    walker_controls = {}  # actor python-id -> carla.WalkerControl

    # Pre-compute camera world transform for visibility check.
    # Camera is at (CAM_X forward, CAM_Z up) relative to ego.
    cam_world_loc = carla.Location(
        x=ego_loc.x + fwd.x * CAM_X,
        y=ego_loc.y + fwd.y * CAM_X,
        z=ego_loc.z + CAM_Z,
    )
    half_fov_rad = math.radians(FOV / 2.0)
    # Margin: require the spawn to be within 80% of the half-FOV to allow
    # some movement before the pedestrian exits the frame.
    fov_margin = 0.80

    def _spawn_is_visible(spawn_loc):
        """Check that spawn_loc projects inside the camera FOV (with margin)."""
        dx = spawn_loc.x - cam_world_loc.x
        dy = spawn_loc.y - cam_world_loc.y
        # Forward distance along camera optical axis
        depth = fwd.x * dx + fwd.y * dy
        if depth < 1.0:
            return False
        # Lateral distance along camera right axis
        lateral = abs(right.x * dx + right.y * dy)
        max_lateral = depth * math.tan(half_fov_rad) * fov_margin
        return lateral < max_lateral

    MAX_SPAWN_ATTEMPTS = 40

    for i in range(n_peds):
        ped = None
        for attempt in range(MAX_SPAWN_ATTEMPTS):
            off_x = np.random.normal(SPAWN_MU[0], max(SPAWN_SIGMA[0], 0.01))
            off_y = np.random.normal(SPAWN_MU[1], max(SPAWN_SIGMA[1], 0.01))

            loc = carla.Location(
                x=ego_loc.x + fwd.x * off_x + right.x * off_y,
                y=ego_loc.y + fwd.y * off_x + right.y * off_y,
                z=ego_loc.z + 0.5,
            )

            if not _spawn_is_visible(loc):
                if attempt < MAX_SPAWN_ATTEMPTS - 1:
                    print(f"  [retry {attempt+1}] Ped {i+1} outside FOV, resampling")
                continue

            wp = world.get_map().get_waypoint(loc)
            if wp:
                loc.z = wp.transform.location.z + 0.5

            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            yaw = random.uniform(0, 360)
            tf = carla.Transform(loc, carla.Rotation(yaw=yaw))
            ped = world.try_spawn_actor(bp, tf)
            if ped is not None:
                break
            if attempt < MAX_SPAWN_ATTEMPTS - 1:
                print(f"  [retry {attempt+1}] Ped {i+1} spawn failed, trying new location")

        if ped is None:
            print(f"  [WARN] Could not spawn pedestrian {i+1} after {MAX_SPAWN_ATTEMPTS} attempts")
            continue

        walkers.append(ped)
        walker_ids[id(ped)] = i + 1

        speed = random.uniform(*PED_SPEED_RANGE)
        dir_yaw = random.uniform(0, 360)
        direction = carla.Vector3D(
            math.cos(math.radians(dir_yaw)),
            math.sin(math.radians(dir_yaw)),
            0.0,
        )
        ctrl = carla.WalkerControl(direction=direction, speed=speed, jump=False)
        walker_controls[id(ped)] = ctrl
        print(f"  Ped {i+1} at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})  "
              f"speed={speed:.2f} m/s  dir_yaw={dir_yaw:.0f}")

    # Store base desired direction per pedestrian (before social force)
    walker_base_dirs = {}   # actor python-id -> carla.Vector3D
    walker_speeds = {}      # actor python-id -> float
    for ped in walkers:
        ctrl = walker_controls[id(ped)]
        walker_base_dirs[id(ped)] = ctrl.direction
        walker_speeds[id(ped)] = ctrl.speed

    print(f"Spawned {len(walkers)}/{n_peds} pedestrians")

    # ── Simulation loop ──────────────────────────────────────────
    dt = 1.0 / FPS
    total_frames = int(SIM_DURATION * FPS)
    frame_idx = 0

    gt_file = open(gt_path, "w", newline="")
    gt_writer = csv.writer(gt_file)
    gt_writer.writerow(["frame", "ped_id", "world_pX", "world_pZ", "world_pY",
                        "bb_left", "bb_top", "bb_w", "bb_h"])

    try:
        for _ in range(total_frames):
            # Compute social force repulsion and apply controls
            locations = {id(p): p.get_location() for p in walkers}
            for ped in walkers:
                pid_key = id(ped)
                base_dir = walker_base_dirs[pid_key]
                loc_i = locations[pid_key]

                repulse_x, repulse_y = 0.0, 0.0
                for other in walkers:
                    if id(other) == pid_key:
                        continue
                    loc_j = locations[id(other)]
                    dx = loc_i.x - loc_j.x
                    dy = loc_i.y - loc_j.y
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                    if dist < SOCIAL_FORCE_RADIUS:
                        strength = SOCIAL_FORCE_STRENGTH * math.exp(
                            -SOCIAL_FORCE_FALLOFF * dist
                        )
                        repulse_x += strength * (dx / dist)
                        repulse_y += strength * (dy / dist)

                # Blend base direction with repulsion
                fx = base_dir.x + repulse_x
                fy = base_dir.y + repulse_y
                mag = math.sqrt(fx * fx + fy * fy) + 1e-8
                final_dir = carla.Vector3D(fx / mag, fy / mag, 0.0)

                ctrl = carla.WalkerControl(
                    direction=final_dir,
                    speed=walker_speeds[pid_key],
                    jump=False,
                )
                ped.apply_control(ctrl)

            world.tick()
            frame_idx += 1

            # Write RGB frame
            rgb_frame = rgb_holder["data"]
            if rgb_frame is not None:
                vid_writer.write(rgb_frame)

            # Write depth frame
            depth_m = depth_holder["data"]
            if depth_m is not None:
                depth_u16 = depth_to_uint16(depth_m, MAX_DEPTH_M)
                png_path = os.path.join(depth_dir, f"depth_{frame_idx:06d}.png")
                cv2.imwrite(png_path, depth_u16)

            # Write GT
            cam_transform = camera_rgb.get_transform()
            for ped in walkers:
                pid = walker_ids[id(ped)]
                row = get_gt_row(ped, pid, cam_transform, K, WIDTH, HEIGHT)
                if row is not None:
                    pX, pZ, pY, bl, bt, bw, bh = row
                    gt_writer.writerow([frame_idx, pid,
                                        f"{pX:.4f}", f"{pZ:.4f}", f"{pY:.4f}",
                                        f"{bl:.2f}", f"{bt:.2f}", f"{bw:.2f}", f"{bh:.2f}"])

            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total_frames}")

    finally:
        gt_file.close()
        camera_rgb.stop();   camera_rgb.destroy()
        camera_depth.stop(); camera_depth.destroy()
        for ped in walkers:
            ped.destroy()
        vehicle.destroy()
        vid_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"Done.  Video -> {os.path.abspath(video_path)}")
        print(f"       Depth -> {os.path.abspath(depth_dir)}/")
        print(f"       GT    -> {os.path.abspath(gt_path)}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARLA experiment scenario recorder")
    parser.add_argument("--n_peds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results/n3")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    args = parser.parse_args()
    run_scenario(args.n_peds, args.seed, args.output_dir, args.host, args.port)


if __name__ == "__main__":
    main()
