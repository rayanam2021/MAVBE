#!/usr/bin/env python
"""
CARLA script: Spawn a stationary ego vehicle, then spawn two pedestrians that
walk toward each other along a crosswalk directly in front of the car.

Behavior:
  - Ped A starts left of frame, walks rightward (east).
  - Ped B starts right of frame, walks leftward (west).
  - Ped B is spawned ~0.6 m further from the camera than Ped A (depth stagger).
  - When they come within AVOID_RADIUS of each other, each nudges slightly
    forward (away from center of the crosswalk) â€” natural avoidance swerve.
  - At the moment they are side-by-side, Ped B is behind Ped A from the
    camera's perspective â†’ brief occlusion.
  - After passing, they resume their original heading and walk off screen.

Output: video (MP4) and ground-truth file (MOT format: frame,id,left,top,width,height)
for use with the perception pipeline (e.g. detect_dual_tracking_kf.py --gt <gt.txt>)
and evaluation (run_metrics.py --gt --pred).

Usage:
    python spawn_ped_corssing.py [--output crosswalk_occlusion.mp4] [--gt-out my_gt.txt]
    python spawn_ped_corssing.py --host 127.0.0.1 --port 2000 --map Town10HD
"""

from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

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

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tuneable parameters
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CROSS_DISTANCE   = 10.0   # metres in front of car where peds walk
HALF_SPAN        =  5.0   # peds start this far left / right of centre
WALK_SPEED       =  1.2   # m/s normal walking speed
AVOID_RADIUS     =  1.8   # metres â€” distance that triggers avoidance swerve
SWERVE_MAGNITUDE =  0.35  # lateral swerve added when avoiding (normalised)
DEPTH_STAGGER    =  0.6   # Ped B is this many metres further from camera
                          # â†’ makes A occlude B when side-by-side
SIM_DURATION     = 20.0   # seconds


def rotate_vector_yaw(vec, yaw_deg):
    """Rotate a 2-D (x, y) vector by yaw_deg degrees (CARLA convention)."""
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return carla.Vector3D(c * vec.x - s * vec.y,
                          s * vec.x + c * vec.y,
                          0.0)


def unit(v):
    """Return a normalised carla.Vector3D (ignore z)."""
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


def build_projection_matrix(width, height, fov_deg):
    fov_rad = math.radians(fov_deg)
    focal = width / (2.0 * math.tan(fov_rad / 2.0))
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
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    return (u, v)


def get_gt_boxes_image(walkers, camera, K, width, height, walker_id_by_actor):
    cam_transform = camera.get_transform()
    boxes = []
    ped_w, ped_h = 50, 150
    for walker in walkers:
        try:
            loc = walker.get_location()
        except Exception:
            continue
        pt = world_to_image(cam_transform, K, loc, width, height)
        if pt is None:
            continue
        u, v = pt
        wid = walker_id_by_actor.get(id(walker))
        if wid is None:
            continue
        left = max(0, min(u - ped_w // 2, width - ped_w))
        top = max(0, min(v - ped_h, height - ped_h))
        boxes.append((wid, left, top, ped_w, ped_h))
    return boxes


def run(args):
    print("Connecting to CARLA at %s:%d ..." % (args.host, args.port), flush=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    print("Connected. Map: %s" % world.get_map().name, flush=True)

    if args.map and not world.get_map().name.endswith(args.map):
        print("Loading map", args.map, "...")
        world = client.load_world(args.map)
        time.sleep(3)

    # â”€â”€ Synchronous mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # â”€â”€ Ego vehicle (stationary) at fixed location â”€â”€â”€â”€â”€â”€â”€â”€
    vehicle_bp = blueprint_library.filter(args.filter)[0]
    vehicle_bp.set_attribute('role_name', 'hero')

    # Fixed spawn: x=-52.310936, y=-1.585238, z=0.600000
    # Yaw=0 means the car faces along the +X axis; adjust if needed.
    spawn_tf = carla.Transform(
        carla.Location(x=-52.310936, y=-1.585238, z=0.600000),
        carla.Rotation(yaw=0.0)
    )
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        print("Could not spawn vehicle at fixed location â€” may be occupied. "
              "Try nudging x/y slightly or clearing existing actors.")
        return

    # Keep vehicle still
    vehicle.set_simulate_physics(False)

    # â”€â”€ Tick once so the server registers the actor's true transform â”€â”€
    world.tick()
    print(f"Ego vehicle spawned at {vehicle.get_transform().location}")

    # â”€â”€ RGB camera â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', '90')
    cam_tf = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0)   # slightly downward â€” good for crosswalk
    )
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    image_holder = {"data": None}

    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        image_holder["data"] = arr[:, :, :3][:, :, ::-1].copy()  # BGRAâ†’BGR

    camera.listen(on_image)

    # â”€â”€ Video writer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.output, fourcc, args.fps, (args.width, args.height)
    )
    gt_path = args.gt_out or (str(Path(args.output).with_suffix('')) + '_gt.txt')
    K = build_projection_matrix(args.width, args.height, 90.0)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Compute crosswalk start positions in world space.
    #
    # The crosswalk runs perpendicular to the car's forward axis:
    #   car_forward  â†’ "into the scene" direction
    #   car_right    â†’ along the crosswalk
    #
    # Ped A  : centre - HALF_SPAN * car_right   (left side)  walks â†’ right
    # Ped B  : centre + HALF_SPAN * car_right   (right side) walks â†’ left
    #           Ped B is DEPTH_STAGGER m further from camera (+ car_forward)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Read transform AFTER the tick so values are populated correctly
    ego_tf  = vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw   # degrees

    # Unit vectors in world frame
    fwd_2d   = carla.Vector3D(math.cos(math.radians(ego_yaw)),
                               math.sin(math.radians(ego_yaw)), 0)
    right_2d = carla.Vector3D(-math.sin(math.radians(ego_yaw)),
                               math.cos(math.radians(ego_yaw)), 0)

    # Base centre of crosswalk
    cross_centre = carla.Location(
        x = ego_loc.x + fwd_2d.x * CROSS_DISTANCE,
        y = ego_loc.y + fwd_2d.y * CROSS_DISTANCE,
        z = ego_loc.z
    )

    # Try to snap z to road surface
    wp = world.get_map().get_waypoint(cross_centre)
    z_ground = wp.transform.location.z + 0.05 if wp else cross_centre.z

    def make_loc(offset_fwd, offset_right):
        return carla.Location(
            x = cross_centre.x + fwd_2d.x * offset_fwd + right_2d.x * offset_right,
            y = cross_centre.y + fwd_2d.y * offset_fwd + right_2d.y * offset_right,
            z = z_ground + 0.5
        )

    # Spawn transforms
    loc_A = make_loc(0.0,          -HALF_SPAN)   # left  of car, no depth offset
    loc_B = make_loc(DEPTH_STAGGER, HALF_SPAN)   # right of car, slightly further

    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    if not walker_bps:
        walker_bps = blueprint_library.filter('walker.*')
    if not walker_bps:
        print("No walker blueprints available.")
        return

    bp_A = random.choice(walker_bps)
    bp_B = random.choice(walker_bps)
    # Make them visually distinct if possible
    while bp_B.id == bp_A.id and len(walker_bps) > 1:
        bp_B = random.choice(walker_bps)

    tf_A = carla.Transform(loc_A, carla.Rotation(yaw=ego_yaw + 90))   # facing right
    tf_B = carla.Transform(loc_B, carla.Rotation(yaw=ego_yaw - 90))   # facing left

    ped_A = world.try_spawn_actor(bp_A, tf_A)
    ped_B = world.try_spawn_actor(bp_B, tf_B)
    if ped_A is None or ped_B is None:
        print("Failed to spawn one or both pedestrians. "
              "Try a different map or adjust CROSS_DISTANCE.")
        if ped_A: ped_A.destroy()
        if ped_B: ped_B.destroy()
        vehicle.destroy()
        camera.stop(); camera.destroy()
        return

    print(f"Ped A spawned at {loc_A}")
    print(f"Ped B spawned at {loc_B}")

    walkers = [ped_A, ped_B]
    walker_id_by_actor = {id(ped_A): 1, id(ped_B): 2}
    gt_file = open(gt_path, 'w')
    frame_idx = 0

    # Base walking directions (world-frame 2-D unit vectors)
    dir_A_base = unit(right_2d)                                # +right
    dir_B_base = unit(carla.Vector3D(-right_2d.x,
                                     -right_2d.y, 0.0))       # âˆ’right
    # Each ped turns to THEIR OWN right when avoiding:
    #   A walks in +right_2d  â†’ their right is +fwd_2d (away from camera)
    #   B walks in -right_2d  â†’ their right is -fwd_2d (toward camera)
    # This makes them diverge like real pedestrians passing on the right.
    swerve_B = unit(fwd_2d)
    swerve_A = unit(carla.Vector3D(-fwd_2d.x, -fwd_2d.y, 0.0))

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Main simulation loop
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sim_time = 0.0
    dt       = 1.0 / args.fps

    # State machine for each ped: 'walk', 'swerve', 'done'
    state = {'A': 'walk', 'B': 'walk'}

    # Track whether pedestrians have passed each other
    passed = False

    try:
        while sim_time < SIM_DURATION:
            world.tick()
            sim_time += dt

            loc_a = ped_A.get_location()
            loc_b = ped_B.get_location()
            dist  = distance_2d(loc_a, loc_b)

            # â”€â”€ Determine movement direction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Ped A
            if dist < AVOID_RADIUS and not passed:
                state['A'] = 'swerve'
                state['B'] = 'swerve'
                dir_A = unit(vec_add(vec_scale(dir_A_base, 1.0),
                                     vec_scale(swerve_A,   SWERVE_MAGNITUDE)))
                dir_B = unit(vec_add(vec_scale(dir_B_base, 1.0),
                                     vec_scale(swerve_B,   SWERVE_MAGNITUDE)))
            else:
                # Once they've crossed (dist growing again), mark passed
                if state['A'] == 'swerve' and dist > AVOID_RADIUS:
                    passed = True
                    state['A'] = 'done'
                    state['B'] = 'done'
                dir_A = dir_A_base
                dir_B = dir_B_base

            # â”€â”€ Apply WalkerControl â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ctrl_A = carla.WalkerControl()
            ctrl_A.direction = dir_A
            ctrl_A.speed     = WALK_SPEED
            ctrl_A.jump      = False
            ped_A.apply_control(ctrl_A)

            ctrl_B = carla.WalkerControl()
            ctrl_B.direction = dir_B
            ctrl_B.speed     = WALK_SPEED
            ctrl_B.jump      = False
            ped_B.apply_control(ctrl_B)

            # â”€â”€ Write video frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            frame = image_holder["data"]
            if frame is not None:
                frame_idx += 1
                gt_boxes = get_gt_boxes_image(
                    walkers, camera, K, args.width, args.height, walker_id_by_actor
                )
                for wid, left, top, pw, ph in gt_boxes:
                    gt_file.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1\n" % (frame_idx, wid, left, top, pw, ph))
                # Overlay HUD
                occlusion_text = ""
                if dist < 0.8:
                    occlusion_text = "[ OCCLUSION ]"
                elif state['A'] == 'swerve':
                    occlusion_text = "[ AVOIDANCE SWERVE ]"

                ts = f"t={sim_time:.1f}s  dist={dist:.2f}m  {occlusion_text}"
                cv2.putText(frame, ts, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                video_writer.write(frame)

    finally:
        gt_file.close()
        camera.stop()
        camera.destroy()
        ped_A.destroy()
        ped_B.destroy()
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Saved video ->", os.path.abspath(args.output))
        print("Saved GT    ->", os.path.abspath(gt_path))


def main():
    parser = argparse.ArgumentParser(
        description="Two pedestrians cross in front of the ego car with occlusion."
    )
    parser.add_argument('--host',   default='127.0.0.1')
    parser.add_argument('--port',   type=int,   default=2000)
    parser.add_argument('--output', default='crosswalk_occlusion.mp4')
    parser.add_argument('--gt-out', default='', help='Ground-truth MOT file path (default: <output_base>_gt.txt)')
    parser.add_argument('--fps',    type=float, default=20.0)
    parser.add_argument('--width',  type=int,   default=1920)
    parser.add_argument('--height', type=int,   default=1080)
    parser.add_argument('--map',    default='Town10HD',
                        help='CARLA map (leave blank to use current)')
    parser.add_argument('--filter', default='vehicle.tesla.model3',
                        help='Ego vehicle blueprint filter')
    args = parser.parse_args()
    print("Output video: %s" % args.output, flush=True)
    gt_out = args.gt_out if args.gt_out else (Path(args.output).with_suffix('').name + '_gt.txt')
    print("Output GT:    %s" % gt_out, flush=True)
    run(args)


if __name__ == "__main__":
    main()
