#!/usr/bin/env python
"""
CARLA scenario: Multiple pedestrians move quickly across the FOV and pass behind
a large occluder (bus/truck), then re-emerge at different times.

Designed to stress-test tracking: appearance-only (e.g. Deep SORT without a good
motion model) can fail (ID swaps, lost tracks) when similar-looking peds emerge
from behind the occluder at different times. A Kalman filter (or behavioral EKF)
helps maintain correct IDs by predicting where each ped will re-emerge.

  - Ego: stationary, forward-facing camera. Weather: snowy/overcast day.
  - Parked car: parked on the side of the road (~5–6 m ahead, road edge, not on
    sidewalk). Pedestrians on the sidewalk walk behind it and get occluded.
  - Peds: 10 pedestrians on sidewalks; some walk along the right sidewalk (pass
    behind the parked car); some cross the road; the rest walk in varied directions.

Usage:
  python occlusion_multi.py --output occlusion_multi.mp4 --duration 30
  python occlusion_multi.py --map Town03 --seed 42

Then run tracking on the video; compare detect_dual_tracking_kf.py (vanilla KF) vs
appearance-only or weak motion model to see ID swaps / lost tracks when peds
re-emerge from behind the occluder at different times.
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


# Nav search cone (in front of car)
NAV_FWD_MIN, NAV_FWD_MAX = 3.0, 15.0
NAV_RIGHT_LO, NAV_RIGHT_HI = -5.0, 5.0
MIN_PEDS_IN_VIEW = 1

# Occlusion scenario geometry (metres, in vehicle frame)
# Parked car: on the ROAD (right edge of lane), not on sidewalk, so peds on sidewalk pass behind it
OCCLUDER_FWD = 5.5       # parked car ~5.5 m ahead
OCCLUDER_RIGHT = 2.5     # on road, right side of lane (sidewalk is further right)
PED_FWD_LO, PED_FWD_HI = 4.0, 14.0
PED_RIGHT_LO, PED_RIGHT_HI = -5.0, 5.0
# Right sidewalk: further right than the parked car; peds walk forward and pass behind car
SIDEWALK_RIGHT_LO, SIDEWALK_RIGHT_HI = 4.5, 6.5
SIDEWALK_FWD_LO, SIDEWALK_FWD_HI = 3.0, 4.5
# Left sidewalk: for crossing peds (spawn here, walk right to cross)
LEFT_SIDEWALK_RIGHT_LO, LEFT_SIDEWALK_RIGHT_HI = -6.5, -4.5
N_SIDEWALK_PEDS = 4      # right sidewalk, walk forward, occluded behind parked car
N_CROSSING_PEDS = 3      # cross the road (e.g. left sidewalk -> walk right)
N_PEDS = 10
PED_SPEED_LO, PED_SPEED_HI = 0.6, 1.3


def _unit_2d(v):
    mag = math.sqrt(v.x ** 2 + v.y ** 2)
    if mag < 1e-6:
        return carla.Vector3D(0, 0, 0)
    return carla.Vector3D(v.x / mag, v.y / mag, 0.0)


def find_spawn_with_peds_in_view(world, spawn_points, n_peds=3, n_nav_samples=250):
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

    def far_enough(loc, others, min_dist=0.8):
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


def _nav_point_in_direction(world, vehicle, min_fwd, max_fwd, min_right, max_right, max_tries=50):
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

    # Snowy/overcast winter day (CARLA has no true snow; use overcast + precipitation + low sun)
    try:
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=50.0,
            precipitation_deposits=80.0,
            wind_intensity=20.0,
            sun_azimuth_angle=15.0,
            sun_altitude_angle=18.0,
            fog_density=15.0,
            fog_distance=50.0,
            wetness=40.0,
        )
        world.set_weather(weather)
        print("[OCCL] Weather set to snowy/overcast winter day.")
    except Exception as e:
        print("[OCCL] Could not set weather:", e)

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
            print("[OCCL] No ped nav in view on current map; loading Town03 (better nav mesh).")
            world = client.load_world("Town03")
            time.sleep(2)
            world = client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / args.fps
            world.apply_settings(settings)
            try:
                world.set_weather(carla.WeatherParameters(cloudiness=90.0, precipitation=50.0, precipitation_deposits=80.0, wind_intensity=20.0, sun_altitude_angle=18.0, fog_density=15.0))
            except Exception:
                pass
            spawn_points = world.get_map().get_spawn_points()
            if spawn_points:
                best_idx, ped_locations = find_spawn_with_peds_in_view(world, spawn_points, n_peds=MIN_PEDS_IN_VIEW)
    if best_idx is None or not ped_locations:
        print("No spawn with pedestrian nav in view after trying Town03.")
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
    vehicle.set_simulate_physics(False)
    print("[OCCL] Ego at spawn index; setting up occluder and pedestrians.")

    v_trans = vehicle.get_transform()
    v_loc = v_trans.location
    v_fwd = v_trans.get_forward_vector()
    v_right = v_trans.get_right_vector()
    v_fwd_u = _unit_2d(v_fwd)
    v_right_u = _unit_2d(v_right)
    yaw = v_trans.rotation.yaw

    # Parked car (occluder): sedan on the right side so sidewalk peds pass behind it
    occluder_bp = None
    for name in ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.tesla.model3", "vehicle.bmw.grandtourer", "vehicle.dodge.charger_police"]:
        try:
            occluder_bp = bp_lib.filter(name)[0]
            break
        except IndexError:
            continue
    if occluder_bp is None:
        occluder_bp = bp_lib.filter("vehicle.*")[0]
    occ_loc = carla.Location(
        v_loc.x + v_fwd.x * OCCLUDER_FWD + v_right.x * OCCLUDER_RIGHT,
        v_loc.y + v_fwd.y * OCCLUDER_FWD + v_right.y * OCCLUDER_RIGHT,
        v_loc.z,
    )
    occ_z = _waypoint_z(world, occ_loc) + 0.5
    occ_loc.z = occ_z
    occ_yaw = yaw  # parallel to road (parked on the side)
    occluder = world.try_spawn_actor(occluder_bp, carla.Transform(occ_loc, carla.Rotation(yaw=occ_yaw)))
    if occluder is not None:
        occluder.set_simulate_physics(False)
        world.tick()
        print("[OCCL] Car parked on side of road (%.1f m ahead, %.1f m right, not on sidewalk)." % (OCCLUDER_FWD, OCCLUDER_RIGHT))
    else:
        print("[OCCL] WARNING: Could not spawn parked car.")

    # Pedestrians: right sidewalk (walk forward, occluded behind car), crossing (left sidewalk -> cross), rest varied
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        print("No walker blueprints.")
        if occluder:
            occluder.destroy()
        vehicle.destroy()
        return

    rng = random.Random(args.seed)
    direction_options = [
        (v_fwd_u, 0.0),
        (_unit_2d(carla.Vector3D(-v_fwd_u.x, -v_fwd_u.y, 0)), 180.0),
        (v_right_u, 90.0),
        (_unit_2d(carla.Vector3D(-v_right_u.x, -v_right_u.y, 0)), -90.0),
    ]

    def make_loc(fwd_off, right_off):
        z = _waypoint_z(world, carla.Location(
            v_loc.x + v_fwd.x * fwd_off + v_right.x * right_off,
            v_loc.y + v_fwd.y * fwd_off + v_right.y * right_off, v_loc.z)) + 0.5
        return carla.Location(
            v_loc.x + v_fwd.x * fwd_off + v_right.x * right_off,
            v_loc.y + v_fwd.y * fwd_off + v_right.y * right_off,
            z,
        )

    walkers = []
    walker_behaviors = []
    n_sidewalk_spawned = 0
    n_crossing_spawned = 0
    for i in range(N_PEDS):
        if i < N_SIDEWALK_PEDS:
            # Right sidewalk: walk forward along sidewalk, pass behind parked car
            spawn_fwd = rng.uniform(SIDEWALK_FWD_LO, SIDEWALK_FWD_HI)
            spawn_right = rng.uniform(SIDEWALK_RIGHT_LO, SIDEWALK_RIGHT_HI)
            nav = _nav_point_in_direction(world, vehicle, SIDEWALK_FWD_LO, SIDEWALK_FWD_HI + 1.0, SIDEWALK_RIGHT_LO - 0.5, SIDEWALK_RIGHT_HI + 0.5)
            direction, face_yaw_offset = (v_fwd_u, 0.0)
        elif i < N_SIDEWALK_PEDS + N_CROSSING_PEDS:
            # Crossing: spawn on left sidewalk, walk right (cross the road)
            spawn_fwd = rng.uniform(SIDEWALK_FWD_LO, PED_FWD_HI * 0.5)
            spawn_right = rng.uniform(LEFT_SIDEWALK_RIGHT_LO, LEFT_SIDEWALK_RIGHT_HI)
            nav = _nav_point_in_direction(world, vehicle, SIDEWALK_FWD_LO, PED_FWD_HI, LEFT_SIDEWALK_RIGHT_LO - 0.5, LEFT_SIDEWALK_RIGHT_HI + 0.5)
            direction, face_yaw_offset = (v_right_u, 90.0)
        else:
            spawn_fwd = rng.uniform(PED_FWD_LO, PED_FWD_HI)
            spawn_right = rng.uniform(PED_RIGHT_LO, PED_RIGHT_HI)
            nav = _nav_point_in_direction(world, vehicle, PED_FWD_LO, PED_FWD_HI, PED_RIGHT_LO, PED_RIGHT_HI)
            direction, face_yaw_offset = direction_options[(i - N_SIDEWALK_PEDS - N_CROSSING_PEDS) % len(direction_options)]

        spawn_loc = nav if nav is not None else make_loc(spawn_fwd, spawn_right)
        if nav is None:
            spawn_loc.z = make_loc(spawn_fwd, spawn_right).z
        bp = walker_bps[rng.randint(0, len(walker_bps) - 1)]
        spawn_yaw = yaw + face_yaw_offset
        rot = carla.Rotation(pitch=0, roll=0, yaw=spawn_yaw)
        w = world.try_spawn_actor(bp, carla.Transform(spawn_loc, rot))
        if w is None:
            w = world.try_spawn_actor(bp, carla.Transform(make_loc(spawn_fwd, spawn_right), rot))
        if w is None:
            continue
        walkers.append(w)
        if i < N_SIDEWALK_PEDS:
            n_sidewalk_spawned += 1
        elif i < N_SIDEWALK_PEDS + N_CROSSING_PEDS:
            n_crossing_spawned += 1
        speed = rng.uniform(PED_SPEED_LO, PED_SPEED_HI)
        walker_behaviors.append({"direction": direction, "speed": speed})
        world.tick()

    print("[OCCL] Spawned %d pedestrians (%d on right sidewalk, %d crossing road)." % (len(walkers), n_sidewalk_spawned, n_crossing_spawned))
    if not walkers:
        print("No pedestrians spawned.")
        if occluder:
            occluder.destroy()
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
    print("[OCCL] Running occlusion scenario (%d peds, %s occluder)." % (len(walkers), "with" if occluder else "no"))

    sim_time = 0.0
    dt = 1.0 / args.fps
    frame_idx = 0
    try:
        while sim_time < args.duration:
            world.tick()
            sim_time += dt
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
        if occluder is not None:
            try:
                occluder.destroy()
            except Exception:
                pass
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Saved video:", args.output)
        print("Saved GT:", gt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occlusion scenario: peds pass behind large occluder at different times.")
    parser.add_argument("--output", default="occlusion_multi.mp4")
    parser.add_argument("--gt-out", default="")
    parser.add_argument("--duration", type=float, default=30.0)
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
