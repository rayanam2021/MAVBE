#!/usr/bin/env python
"""
CARLA script: Spawn a stationary ego vehicle, then spawn two pedestrians that
walk toward each other along a crosswalk directly in front of the car.

Now saves TWO videos in parallel:
  - <output>            : standard RGB video  (e.g. crosswalk_occlusion.mp4)
  - <output>_depth.mp4  : per-pixel depth video

Depth encoding in the depth video
──────────────────────────────────
CARLA's sensor.camera.depth returns a BGRA image whose R, G, B channels
encode depth in metres using a 24-bit fixed-point scheme:

    depth_m = (R + G*256 + B*65536) / (256³ − 1) × 1000

This script decodes that value and writes two depth representations:

  1. 8-bit grayscale video  — pixel = clamp(depth_m / MAX_DEPTH_M × 255, 0, 255)
     Easy to play back; precision is ~(MAX_DEPTH_M / 255) metres per step.

  2. 16-bit PNG sequence    — saved alongside the video as depth_frames/*.png
     Each pixel holds an uint16 value where 1 LSB ≈ MAX_DEPTH_M / 65535 metres.
     Full lossless precision; use these if you need exact depth values.

Set MAX_DEPTH_M to the maximum depth you expect in the scene (default 50 m).

Usage:
    python crosswalk_occlusion.py [--host 127.0.0.1] [--port 2000]
                                   [--output occlusion.mp4] [--fps 20]
                                   [--map Town10HD]
"""

from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time

try:
    import numpy as np
    import cv2
except ImportError:
    print("pip install numpy opencv-python")
    sys.exit(1)

try:
    import carla
except ImportError:
    print("CARLA module not found. Add CARLA PythonAPI to PYTHONPATH.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# Tuneable parameters
# ─────────────────────────────────────────────────────────
CROSS_DISTANCE   = 10.0   # metres in front of car where peds walk
HALF_SPAN        =  5.0   # peds start this far left / right of centre
WALK_SPEED       =  1.2   # m/s normal walking speed
AVOID_RADIUS     =  1.8   # metres — distance that triggers avoidance swerve
SWERVE_MAGNITUDE =  0.35  # lateral swerve added when avoiding (normalised)
DEPTH_STAGGER    =  0.6   # Ped B is this many metres further from camera
SIM_DURATION     = 20.0   # seconds

MAX_DEPTH_M      = 50.0   # metres — depth values are normalised against this


# ─────────────────────────────────────────────────────────
# Depth decoding helpers
# ─────────────────────────────────────────────────────────

def decode_depth_metres(bgra_img: np.ndarray) -> np.ndarray:
    """
    Convert a CARLA depth image (BGRA uint8, shape H×W×4) to a float32
    array of depth values in metres (shape H×W).

    CARLA packs depth into the R, G, B channels:
        depth_normalised = (R + G*256 + B*65536) / (256³ − 1)
        depth_metres     = depth_normalised × 1000
    OpenCV loads channels as B G R A, so:
        R = channel 2,  G = channel 1,  B = channel 0
    """
    R = bgra_img[:, :, 2].astype(np.float32)
    G = bgra_img[:, :, 1].astype(np.float32)
    B = bgra_img[:, :, 0].astype(np.float32)
    normalised = (R + G * 256.0 + B * 65536.0) / (256.0 ** 3 - 1.0)
    return normalised * 1000.0   # → metres


def depth_to_uint8(depth_m: np.ndarray, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """
    Map depth (metres, float32) → 8-bit greyscale.
    Closer objects are brighter (0 m → 255, max_depth → 0).
    This is the inverse-depth convention used by many visualisers.
    """
    # Clamp
    d = np.clip(depth_m, 0.0, max_depth)
    # Invert so near = bright, then scale to 0-255
    grey = ((1.0 - d / max_depth) * 255.0).astype(np.uint8)
    return grey


def depth_to_uint16(depth_m: np.ndarray, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """
    Map depth (metres, float32) → 16-bit unsigned integer.
    1 LSB = max_depth / 65535 metres  (≈ 0.76 mm at MAX_DEPTH_M = 50 m).
    """
    d = np.clip(depth_m, 0.0, max_depth)
    return (d / max_depth * 65535.0).astype(np.uint16)


def depth_to_colormap(depth_m: np.ndarray, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Return a BGR colour-mapped (COLORMAP_TURBO) depth image for visual inspection."""
    grey8 = depth_to_uint8(depth_m, max_depth)
    return cv2.applyColorMap(grey8, cv2.COLORMAP_TURBO)


# ─────────────────────────────────────────────────────────
# Geometry helpers (unchanged)
# ─────────────────────────────────────────────────────────

def unit(v):
    mag = math.sqrt(v.x ** 2 + v.y ** 2)
    if mag < 1e-6:
        return carla.Vector3D(0, 0, 0)
    return carla.Vector3D(v.x / mag, v.y / mag, 0.0)

def vec_add(a, b):
    return carla.Vector3D(a.x + b.x, a.y + b.y, 0.0)

def vec_scale(v, s):
    return carla.Vector3D(v.x * s, v.y * s, 0.0)

def distance_2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world  = client.get_world()

    if args.map and not world.get_map().name.endswith(args.map):
        print("Loading map", args.map, "...")
        world = client.load_world(args.map)
        time.sleep(3)

    # ── Synchronous mode ──────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # ── Ego vehicle ───────────────────────────────────────
    vehicle_bp = blueprint_library.filter(args.filter)[0]
    vehicle_bp.set_attribute('role_name', 'hero')

    spawn_tf = carla.Transform(
        carla.Location(x=-52.310936, y=-1.585238, z=0.600000),
        carla.Rotation(yaw=0.0)
    )
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        print("Could not spawn vehicle at fixed location.")
        return

    vehicle.set_simulate_physics(False)
    world.tick()
    print(f"Ego vehicle spawned at {vehicle.get_transform().location}")

    # ── Camera mount transform (shared by both sensors) ───
    cam_tf = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0)
    )

    # ── RGB camera ────────────────────────────────────────
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', '90')
    camera_rgb = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    rgb_holder = {"data": None}
    def on_rgb(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        rgb_holder["data"] = arr[:, :, :3][:, :, ::-1].copy()   # BGRA→BGR

    camera_rgb.listen(on_rgb)

    # ── Depth camera ──────────────────────────────────────
    # sensor.camera.depth produces a 24-bit fixed-point depth map
    # packed into an BGRA image.  It shares the same pose as the RGB camera
    # so the depth map is pixel-perfectly aligned with the colour image.
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(args.width))
    depth_bp.set_attribute('image_size_y', str(args.height))
    depth_bp.set_attribute('fov', '90')
    camera_depth = world.spawn_actor(depth_bp, cam_tf, attach_to=vehicle)

    depth_holder = {"data": None}   # float32 H×W array in metres
    def on_depth(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        depth_holder["data"] = decode_depth_metres(arr)

    camera_depth.listen(on_depth)

    # ── Output paths ──────────────────────────────────────
    base, ext      = os.path.splitext(os.path.abspath(args.output))
    depth_vid_path = base + "_depth" + (ext or ".mp4")
    depth_png_dir  = base + "_depth_frames"
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    os.makedirs(depth_png_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # RGB video (colour, 3-channel)
    rgb_writer = cv2.VideoWriter(
        args.output, fourcc, args.fps, (args.width, args.height)
    )
    # Depth video: colour-mapped turbo for intuitive viewing
    # (near = warm, far = cool)
    depth_writer = cv2.VideoWriter(
        depth_vid_path, fourcc, args.fps, (args.width, args.height)
    )

    # ── Pedestrian setup (unchanged logic) ────────────────
    ego_tf  = vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw

    fwd_2d   = carla.Vector3D(math.cos(math.radians(ego_yaw)),
                               math.sin(math.radians(ego_yaw)), 0)
    right_2d = carla.Vector3D(-math.sin(math.radians(ego_yaw)),
                               math.cos(math.radians(ego_yaw)), 0)

    cross_centre = carla.Location(
        x=ego_loc.x + fwd_2d.x * CROSS_DISTANCE,
        y=ego_loc.y + fwd_2d.y * CROSS_DISTANCE,
        z=ego_loc.z
    )
    wp = world.get_map().get_waypoint(cross_centre)
    z_ground = wp.transform.location.z + 0.05 if wp else cross_centre.z

    def make_loc(offset_fwd, offset_right):
        return carla.Location(
            x=cross_centre.x + fwd_2d.x * offset_fwd + right_2d.x * offset_right,
            y=cross_centre.y + fwd_2d.y * offset_fwd + right_2d.y * offset_right,
            z=z_ground + 0.5
        )

    loc_A = make_loc(0.0,          -HALF_SPAN)
    loc_B = make_loc(DEPTH_STAGGER, HALF_SPAN)

    walker_bps = blueprint_library.filter('walker.pedestrian.*') or \
                 blueprint_library.filter('walker.*')
    if not walker_bps:
        print("No walker blueprints available.")
        return

    bp_A = random.choice(walker_bps)
    bp_B = random.choice(walker_bps)
    while bp_B.id == bp_A.id and len(walker_bps) > 1:
        bp_B = random.choice(walker_bps)

    tf_A = carla.Transform(loc_A, carla.Rotation(yaw=ego_yaw + 90))
    tf_B = carla.Transform(loc_B, carla.Rotation(yaw=ego_yaw - 90))

    ped_A = world.try_spawn_actor(bp_A, tf_A)
    ped_B = world.try_spawn_actor(bp_B, tf_B)
    if ped_A is None or ped_B is None:
        print("Failed to spawn pedestrians.")
        for a in [ped_A, ped_B, vehicle]:
            if a: a.destroy()
        camera_rgb.stop();   camera_rgb.destroy()
        camera_depth.stop(); camera_depth.destroy()
        return

    print(f"Ped A spawned at {loc_A}")
    print(f"Ped B spawned at {loc_B}")
    print(f"RGB  video  → {args.output}")
    print(f"Depth video → {depth_vid_path}")
    print(f"Depth PNGs  → {depth_png_dir}/")

    dir_A_base = unit(right_2d)
    dir_B_base = unit(carla.Vector3D(-right_2d.x, -right_2d.y, 0.0))
    swerve_B   = unit(fwd_2d)
    swerve_A   = unit(carla.Vector3D(-fwd_2d.x, -fwd_2d.y, 0.0))

    # ── Simulation loop ───────────────────────────────────
    sim_time = 0.0
    dt       = 1.0 / args.fps
    state    = {'A': 'walk', 'B': 'walk'}
    passed   = False
    frame_idx = 0

    try:
        while sim_time < SIM_DURATION:
            world.tick()
            sim_time += dt
            frame_idx += 1

            loc_a = ped_A.get_location()
            loc_b = ped_B.get_location()
            dist  = distance_2d(loc_a, loc_b)

            # ── Pedestrian direction logic ─────────────────
            if dist < AVOID_RADIUS and not passed:
                state['A'] = 'swerve'
                state['B'] = 'swerve'
                dir_A = unit(vec_add(vec_scale(dir_A_base, 1.0),
                                     vec_scale(swerve_A,   SWERVE_MAGNITUDE)))
                dir_B = unit(vec_add(vec_scale(dir_B_base, 1.0),
                                     vec_scale(swerve_B,   SWERVE_MAGNITUDE)))
            else:
                if state['A'] == 'swerve' and dist > AVOID_RADIUS:
                    passed = True
                    state['A'] = 'done'
                    state['B'] = 'done'
                dir_A = dir_A_base
                dir_B = dir_B_base

            ctrl_A = carla.WalkerControl(direction=dir_A, speed=WALK_SPEED, jump=False)
            ctrl_B = carla.WalkerControl(direction=dir_B, speed=WALK_SPEED, jump=False)
            ped_A.apply_control(ctrl_A)
            ped_B.apply_control(ctrl_B)

            # ── HUD overlay helper ─────────────────────────
            label = ("[ OCCLUSION ]"        if dist < 0.8 else
                     "[ AVOIDANCE SWERVE ]" if state['A'] == 'swerve' else "")
            ts = f"t={sim_time:.1f}s  dist={dist:.2f}m  {label}"

            # ── Write RGB frame ────────────────────────────
            rgb_frame = rgb_holder["data"]
            if rgb_frame is not None:
                cv2.putText(rgb_frame, ts, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                rgb_writer.write(rgb_frame)

            # ── Write depth frame ──────────────────────────
            depth_m = depth_holder["data"]
            if depth_m is not None:
                # ① Colour-mapped video  (turbo: near=red/warm, far=blue/cool)
                depth_colour = depth_to_colormap(depth_m, MAX_DEPTH_M)
                # Overlay same HUD in white for readability on colormap
                cv2.putText(depth_colour, ts, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                # Add a scale bar legend in the corner
                _add_depth_legend(depth_colour, MAX_DEPTH_M)
                depth_writer.write(depth_colour)

                # ② Lossless 16-bit PNG  (preserves full floating-point precision)
                depth_u16 = depth_to_uint16(depth_m, MAX_DEPTH_M)
                png_path  = os.path.join(depth_png_dir, f"depth_{frame_idx:06d}.png")
                cv2.imwrite(png_path, depth_u16)

    finally:
        camera_rgb.stop();   camera_rgb.destroy()
        camera_depth.stop(); camera_depth.destroy()
        ped_A.destroy()
        ped_B.destroy()
        vehicle.destroy()
        rgb_writer.release()
        depth_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"Done.  RGB video  → {os.path.abspath(args.output)}")
        print(f"       Depth video → {os.path.abspath(depth_vid_path)}")
        print(f"       Depth PNGs  → {os.path.abspath(depth_png_dir)}/")
        print(f"       Depth scale: 1 uint16 LSB = "
              f"{MAX_DEPTH_M / 65535 * 1000:.2f} mm")


def _add_depth_legend(img: np.ndarray, max_depth: float) -> None:
    """
    Draw a small horizontal colour-bar legend with 'NEAR' and 'FAR' labels
    in the bottom-right corner of the depth colour-map image.
    """
    h, w = img.shape[:2]
    bar_w, bar_h = 200, 18
    margin = 12

    # Build a 1×256 turbo colourmap strip and resize
    strip = np.arange(256, dtype=np.uint8).reshape(1, 256)
    strip_colour = cv2.applyColorMap(strip, cv2.COLORMAP_TURBO)  # 1×256×3
    bar = cv2.resize(strip_colour, (bar_w, bar_h))

    x0 = w - bar_w - margin
    y0 = h - bar_h - margin - 20   # leave room for text below

    img[y0:y0 + bar_h, x0:x0 + bar_w] = bar

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "NEAR",       (x0,           y0 + bar_h + 14), font, 0.45, (255,255,255), 1)
    cv2.putText(img, "FAR",        (x0 + bar_w - 28, y0 + bar_h + 14), font, 0.45, (255,255,255), 1)
    cv2.putText(img, f"0 – {max_depth:.0f} m",
                                   (x0 + bar_w // 2 - 28, y0 - 4),  font, 0.45, (200,200,200), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Two pedestrians cross in front of the ego car — saves RGB + depth video."
    )
    parser.add_argument('--host',   default='127.0.0.1')
    parser.add_argument('--port',   type=int,   default=2000)
    parser.add_argument('--output', default='crosswalk_occlusion_with_depth.mp4')
    parser.add_argument('--fps',    type=float, default=20.0)
    parser.add_argument('--width',  type=int,   default=1920)
    parser.add_argument('--height', type=int,   default=1080)
    parser.add_argument('--map',    default='Town10HD',
                        help='CARLA map (leave blank to use current)')
    parser.add_argument('--filter', default='vehicle.tesla.model3',
                        help='Ego vehicle blueprint filter')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()