#!/usr/bin/env python
"""
CARLA script: Spawn a stationary ego vehicle, then spawn two pedestrians that
walk toward each other along a crosswalk directly in front of the car.

Behavior:
  - Ped A starts left of frame, walks rightward (east).
  - Ped B starts right of frame, walks leftward (west).
  - Ped B is spawned ~0.6 m further from the camera than Ped A (depth stagger).
  - When they come within AVOID_RADIUS of each other, each nudges slightly
    forward (away from center of the crosswalk) — natural avoidance swerve.
  - At the moment they are side-by-side, Ped B is behind Ped A from the
    camera's perspective → brief occlusion.
  - After passing, they resume their original heading and walk off screen.

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
                          # → makes A occlude B when side-by-side
SIM_DURATION     = 25.0   # seconds


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

    # ── Ego vehicle (stationary) at fixed location ────────
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
        print("Could not spawn vehicle at fixed location — may be occupied. "
              "Try nudging x/y slightly or clearing existing actors.")
        return

    # Keep vehicle still
    vehicle.set_simulate_physics(False)

    # ── Tick once so the server registers the actor's true transform ──
    world.tick()
    print(f"Ego vehicle spawned at {vehicle.get_transform().location}")

    # ── RGB camera ────────────────────────────────────────
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', '90')
    cam_tf = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-5.0)   # slightly downward — good for crosswalk
    )
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    image_holder = {"data": None}

    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))
        image_holder["data"] = arr[:, :, :3][:, :, ::-1].copy()  # BGRA→BGR

    camera.listen(on_image)

    # ── Video writer ──────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.output, fourcc, args.fps, (args.width, args.height)
    )

    # ─────────────────────────────────────────────────────
    # Compute crosswalk start positions in world space.
    #
    # The crosswalk runs perpendicular to the car's forward axis:
    #   car_forward  → "into the scene" direction
    #   car_right    → along the crosswalk
    #
    # Ped A  : centre - HALF_SPAN * car_right   (left side)  walks → right
    # Ped B  : centre + HALF_SPAN * car_right   (right side) walks → left
    #           Ped B is DEPTH_STAGGER m further from camera (+ car_forward)
    # ─────────────────────────────────────────────────────
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

    # Base walking directions (world-frame 2-D unit vectors)
    dir_A_base = unit(right_2d)                                # +right
    dir_B_base = unit(carla.Vector3D(-right_2d.x,
                                     -right_2d.y, 0.0))       # −right
    # Forward swerve direction for each ped (away from camera = +fwd)
    # swerve_A = unit(fwd_2d)   # A swerves slightly forward (away from cam)
    # swerve_B = unit(-fwd_2d)   # B also swerves forward — keeps them on same side

    swerve_A = unit(fwd_2d)
    swerve_B = unit(carla.Vector3D(-fwd_2d.x, fwd_2d.y, 0.0))

    # ─────────────────────────────────────────────────────
    # Main simulation loop
    # ─────────────────────────────────────────────────────
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

            # ── Determine movement direction ───────────────
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

            # ── Apply WalkerControl ────────────────────────
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

            # ── Write video frame ──────────────────────────
            frame = image_holder["data"]
            if frame is not None:
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
        camera.stop()
        camera.destroy()
        ped_A.destroy()
        ped_B.destroy()
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"Saved video → {os.path.abspath(args.output)}")


def main():
    parser = argparse.ArgumentParser(
        description="Two pedestrians cross in front of the ego car with occlusion."
    )
    parser.add_argument('--host',   default='127.0.0.1')
    parser.add_argument('--port',   type=int,   default=2000)
    parser.add_argument('--output', default='crosswalk_occlusion.mp4')
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