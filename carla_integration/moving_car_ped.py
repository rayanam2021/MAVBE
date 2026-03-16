#!/usr/bin/env python
"""
CARLA scenario: Moving car (following road rules via TrafficManager) with
pedestrians on the sidewalk near a crosswalk. A couple of pedestrians have
a high likelihood to cross the street.

  - Car: drives with TrafficManager (obey lights, speed limit, no lane change).
  - Peds: 5 pedestrians on the sidewalk near a crosswalk; 2 are "crossers" who
    start on the sidewalk and after a short delay cross the road (then return to
    sidewalk behavior). The other 3 walk along the sidewalk.
  - Camera: driver view; video and ground-truth are recorded.

Usage:
  python moving_car_ped.py --output moving_car_ped.mp4 --duration 40
  python moving_car_ped.py --map Town03 --seed 42
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:
    print("CARLA module not found. Add CARLA PythonAPI to PYTHONPATH.")
    sys.exit(1)


# Nav search cone (in front of car) for spawn selection
NAV_FWD_MIN, NAV_FWD_MAX = 3.0, 18.0
NAV_RIGHT_LO, NAV_RIGHT_HI = -6.0, 6.0
MIN_PEDS_IN_VIEW = 1

# Sidewalk and crosswalk (vehicle frame, metres)
# Sidewalk on one side: peds spawn here and walk along it; crossers eventually cross
SIDEWALK_RIGHT_LO, SIDEWALK_RIGHT_HI = 4.0, 6.5   # right sidewalk
SIDEWALK_FWD_LO, SIDEWALK_FWD_HI = 28.0, 55.0     # sidewalk ahead of car (a bit closer than before)
N_PEDS = 5
N_CROSSERS = 2       # these have high likelihood to cross (they will cross after a delay)
PED_SPEED_LO, PED_SPEED_HI = 0.95, 1.55             # faster pedestrians
# Crossers: start crossing after this many seconds (random in range), cross for this long
CROSS_START_LO, CROSS_START_HI = 3.0, 10.0
CROSS_DURATION = 4.5

# Constant-speed driving (no TM speed oscillation): target speed in m/s, gentle throttle/brake
TARGET_SPEED_MS = 3.0
THROTTLE_GAIN = 0.18
BRAKE_GAIN = 0.25
STEER_LOOKAHEAD = 4.0
STEER_GAIN = 2.5


def _unit_2d(v):
    mag = math.sqrt(v.x ** 2 + v.y ** 2)
    if mag < 1e-6:
        return carla.Vector3D(0, 0, 0)
    return carla.Vector3D(v.x / mag, v.y / mag, 0.0)


def find_spawn_with_peds_in_view(world, spawn_points, n_peds=1, n_nav_samples=300):
    nav_points = []
    for _ in range(n_nav_samples):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            nav_points.append(loc)
    if not nav_points:
        return None, []

    def in_front(sp_tf, loc):
        sp_loc = sp_tf.location
        fwd = sp_tf.get_forward_vector()
        right = sp_tf.get_right_vector()
        dx, dy = loc.x - sp_loc.x, loc.y - sp_loc.y
        fwd_d = dx * fwd.x + dy * fwd.y
        right_d = dx * right.x + dy * right.y
        return NAV_FWD_MIN <= fwd_d <= NAV_FWD_MAX and NAV_RIGHT_LO <= right_d <= NAV_RIGHT_HI

    def far_enough(loc, others, min_dist=0.9):
        for o in others:
            if math.hypot(loc.x - o.x, loc.y - o.y) < min_dist:
                return False
        return True

    best_idx, best_list = None, []
    for idx, sp_tf in enumerate(spawn_points):
        in_front_list = []
        for np in nav_points:
            if not in_front(sp_tf, np) or not far_enough(np, in_front_list):
                continue
            in_front_list.append(np)
            if len(in_front_list) >= n_peds:
                break
        if len(in_front_list) > len(best_list):
            best_list = in_front_list[:n_peds]
            best_idx = idx
        if len(best_list) >= n_peds:
            break
    return best_idx, best_list


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


def _nav_point_in_direction(world, vehicle, min_fwd, max_fwd, min_right, max_right, max_tries=80):
    v_loc = vehicle.get_location()
    v_fwd = vehicle.get_transform().get_forward_vector()
    v_right = vehicle.get_transform().get_right_vector()
    for _ in range(max_tries):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        dx = loc.x - v_loc.x
        dy = loc.y - v_loc.y
        fwd_d = dx * v_fwd.x + dy * v_fwd.y
        right_d = dx * v_right.x + dy * v_right.y
        if min_fwd <= fwd_d <= max_fwd and min_right <= right_d <= max_right:
            return loc
    return None


def _waypoint_z(world, loc):
    wp = world.get_map().get_waypoint(loc)
    if wp:
        return wp.transform.location.z
    return loc.z


def run_scenario(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if args.map and not world.get_map().name.endswith(args.map):
        world = client.load_world(args.map)
        time.sleep(2)
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    random.seed(args.seed)
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter(args.filter)[0]
    vehicle_bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points.")
        return

    best_idx, ped_locations = find_spawn_with_peds_in_view(world, spawn_points, n_peds=MIN_PEDS_IN_VIEW)
    if best_idx is None or not ped_locations:
        if not args.map or world.get_map().name.endswith("Town10HD"):
            print("[MCP] No ped nav in view; loading Town03.")
            world = client.load_world("Town03")
            time.sleep(2)
            world = client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / args.fps
            world.apply_settings(settings)
            spawn_points = world.get_map().get_spawn_points()
            if spawn_points:
                best_idx, ped_locations = find_spawn_with_peds_in_view(world, spawn_points, n_peds=MIN_PEDS_IN_VIEW)
    if best_idx is None or not ped_locations:
        print("No spawn with pedestrian nav in view.")
        return

    vehicle = None
    for i in range(len(spawn_points)):
        idx = (best_idx + i) % len(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[idx])
        if vehicle is not None:
            break
    if vehicle is None:
        print("No free vehicle spawn.")
        return
    world.tick()

    # Constant-speed control (no TrafficManager) to avoid brake/accelerate oscillation
    traffic_manager = None
    use_constant_speed = True
    if use_constant_speed:
        print("[MCP] Car using constant-speed control (%.1f m/s, waypoint following)." % TARGET_SPEED_MS)
    else:
        try:
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 50)
            traffic_manager.auto_lane_change(vehicle, False)
            print("[MCP] Car using TrafficManager.")
        except Exception as e:
            print("[MCP] TrafficManager not available (%s); car stationary." % e)

    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    v_fwd = v_trans.get_forward_vector()
    v_right = v_trans.get_right_vector()
    v_fwd_u = _unit_2d(v_fwd)
    v_right_u = _unit_2d(v_right)
    yaw = v_trans.rotation.yaw

    def make_loc(fwd_off, right_off):
        z = _waypoint_z(world, carla.Location(
            v_loc.x + v_fwd.x * fwd_off + v_right.x * right_off,
            v_loc.y + v_fwd.y * fwd_off + v_right.y * right_off, v_loc.z)) + 0.5
        return carla.Location(
            v_loc.x + v_fwd.x * fwd_off + v_right.x * right_off,
            v_loc.y + v_fwd.y * fwd_off + v_right.y * right_off,
            z,
        )

    # Pedestrians on sidewalk near crosswalk; first N_CROSSERS will cross the street
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        print("No walker blueprints.")
        vehicle.destroy()
        return

    rng = random.Random(args.seed)
    walkers = []
    walker_behaviors = []  # list of dicts: direction, speed, is_crosser, state, cross_start_t, cross_end_t

    for i in range(N_PEDS):
        is_crosser = i < N_CROSSERS
        spawn_fwd = rng.uniform(SIDEWALK_FWD_LO, SIDEWALK_FWD_HI)
        spawn_right = rng.uniform(SIDEWALK_RIGHT_LO, SIDEWALK_RIGHT_HI)
        nav = _nav_point_in_direction(world, vehicle, SIDEWALK_FWD_LO, SIDEWALK_FWD_HI + 2.0,
                                      SIDEWALK_RIGHT_LO - 0.5, SIDEWALK_RIGHT_HI + 0.5)
        spawn_loc = nav if nav is not None else make_loc(spawn_fwd, spawn_right)
        if nav is None:
            spawn_loc.z = make_loc(spawn_fwd, spawn_right).z

        bp = walker_bps[rng.randint(0, len(walker_bps) - 1)]
        # Along sidewalk = forward (same as road direction)
        spawn_yaw = yaw
        rot = carla.Rotation(pitch=0, roll=0, yaw=spawn_yaw)
        w = world.try_spawn_actor(bp, carla.Transform(spawn_loc, rot))
        if w is None:
            w = world.try_spawn_actor(bp, carla.Transform(make_loc(spawn_fwd, spawn_right), rot))
        if w is None:
            continue

        speed = rng.uniform(PED_SPEED_LO, PED_SPEED_HI)
        # Crossers: after a random delay, walk left (across road); then back to sidewalk
        if is_crosser:
            cross_start_t = rng.uniform(CROSS_START_LO, CROSS_START_HI)
            walker_behaviors.append({
                "direction": v_fwd_u,
                "speed": speed,
                "is_crosser": True,
                "state": "sidewalk",
                "cross_start_t": cross_start_t,
                "cross_end_t": cross_start_t + CROSS_DURATION,
                "sidewalk_direction": v_fwd_u,
                "cross_direction": _unit_2d(carla.Vector3D(-v_right.x, -v_right.y, 0)),  # left (toward road)
            })
        else:
            # Slight random forward/back along sidewalk
            if rng.random() < 0.5:
                sidewalk_dir = v_fwd_u
            else:
                sidewalk_dir = _unit_2d(carla.Vector3D(-v_fwd_u.x, -v_fwd_u.y, 0))
            walker_behaviors.append({
                "direction": sidewalk_dir,
                "speed": speed,
                "is_crosser": False,
                "state": "sidewalk",
                "cross_start_t": None,
                "cross_end_t": None,
                "sidewalk_direction": sidewalk_dir,
                "cross_direction": None,
            })
        walkers.append(w)
        world.tick()

    print("[MCP] Spawned %d pedestrians (%d will cross the street)." % (len(walkers), min(N_CROSSERS, len(walkers))))
    if not walkers:
        vehicle.destroy()
        return

    # Camera
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=-10))
    camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)
    image_queue = []

    def on_image(im):
        arr = np.frombuffer(im.raw_data, dtype=np.uint8).reshape((im.height, im.width, 4))
        image_queue.append(arr[:, :, :3][:, :, ::-1].copy())

    camera.listen(on_image)
    K = build_projection_matrix(args.width, args.height, 90.0)
    walker_id_by_actor = {id(w): i + 1 for i, w in enumerate(walkers)}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    gt_path = args.gt_out or (str(Path(args.output).with_suffix("")) + "_gt.txt")
    gt_file = open(gt_path, "w")

    for _ in range(5):
        world.tick()
    while image_queue:
        image_queue.pop(0)
    print("[MCP] Recording (car moving, peds on sidewalk / crossing).")

    sim_time = 0.0
    dt = 1.0 / args.fps
    frame_idx = 0
    try:
        while sim_time < args.duration:
            world.tick()
            sim_time += dt

            # Constant-speed vehicle control (waypoint following + speed regulator)
            if use_constant_speed:
                v = vehicle.get_velocity()
                speed_ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
                if speed_ms < TARGET_SPEED_MS:
                    throttle, brake = THROTTLE_GAIN, 0.0
                else:
                    throttle, brake = 0.0, BRAKE_GAIN
                steer = 0.0
                try:
                    wp = world.get_map().get_waypoint(vehicle.get_location())
                    next_wps = wp.next(STEER_LOOKAHEAD)
                    if next_wps:
                        target_loc = next_wps[0].transform.location
                        to_target = carla.Vector3D(
                            target_loc.x - vehicle.get_location().x,
                            target_loc.y - vehicle.get_location().y,
                            0.0,
                        )
                        fwd = vehicle.get_transform().get_forward_vector()
                        dot = fwd.x * to_target.x + fwd.y * to_target.y
                        cross = fwd.x * to_target.y - fwd.y * to_target.x
                        angle_rad = math.atan2(cross, dot) if (dot * dot + cross * cross) > 1e-6 else 0.0
                        steer = max(-1.0, min(1.0, angle_rad * STEER_GAIN))
                except Exception:
                    pass
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

            # Update crosser state and set direction/speed for each ped
            for bhv in walker_behaviors:
                if not bhv["is_crosser"]:
                    continue
                if bhv["state"] == "sidewalk" and sim_time >= bhv["cross_start_t"]:
                    bhv["state"] = "crossing"
                    bhv["direction"] = bhv["cross_direction"]
                elif bhv["state"] == "crossing" and sim_time >= bhv["cross_end_t"]:
                    bhv["state"] = "sidewalk"
                    bhv["direction"] = bhv["sidewalk_direction"]

            for w, bhv in zip(walkers, walker_behaviors):
                ctrl = carla.WalkerControl()
                ctrl.direction = bhv["direction"]
                ctrl.speed = bhv["speed"]
                ctrl.jump = False
                w.apply_control(ctrl)

            while image_queue:
                frame = image_queue.pop(0)
                frame_idx += 1
                gt_boxes = get_gt_boxes_image(walkers, camera, K, args.width, args.height, walker_id_by_actor)
                for wid, left, top, pw, ph in gt_boxes:
                    gt_file.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1\n" % (frame_idx, wid, left, top, pw, ph))
                video_writer.write(frame)
    finally:
        gt_file.close()
        camera.stop()
        camera.destroy()
        for w in walkers:
            try:
                w.destroy()
            except Exception:
                pass
        if traffic_manager is not None:
            try:
                vehicle.set_autopilot(False)
            except Exception:
                pass
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Saved video:", args.output)
        print("Saved GT:", gt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moving car + pedestrians on sidewalk near crosswalk; some cross.")
    parser.add_argument("--output", default="moving_car_ped.mp4")
    parser.add_argument("--gt-out", default="")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--filter", default="vehicle.tesla.model3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_scenario(args)
