#!/usr/bin/env python
"""
CARLA script: Multi-pedestrian scenario with three AI pedestrians.
View is the car's forward-facing camera (driver view). Peds cross the road in front of the car.

  - Car: script tries map spawn points until one has 3+ pedestrian nav points in front (4-12 m), so peds are always in view.
  - Camera: driver view; peds spawn 4-12 m ahead, lateral ±3.5 m (in FOV).
  - Peds: (1) one walks away from the car, (2) one on the other side walks toward the car, (3) one crosses the street.
  - Random (seeded): each ped's speed and heading vary per run.

Car: --stationary (default): car does not move. Without --stationary: TrafficManager (obey lights).

Usage:
  python scenario_pedestrian_crossing_multi.py --output out.mp4 --duration 25
  python scenario_pedestrian_crossing_multi.py --seed 42 --spawn-index 0
  --no-stationary  # car drives with TM
  --no-filter      # use ground-truth for collision check
"""

import argparse
import os
import sys
import random
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:
    print("CARLA module not found. Add CARLA PythonAPI to PYTHONPATH or place it in MAVBE/carla/")
    sys.exit(1)

# Optional: perception filter for collision prediction (repo root must be on path)
FILTER_AVAILABLE = False
try:
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from carla_integration.mot_perception import create_mot_state, run_frame
    FILTER_AVAILABLE = True
except Exception:
    pass


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


def danger_zone_collision_prediction(track_boxes, width, height):
    """
    Predict collision if any track (x1,y1,x2,y2) has center in the danger zone:
    center-x within middle 30% of image, center-y in lower 55% (in front of car).
    track_boxes: list of (x1, y1, x2, y2) or (x1, y1, x2, y2, tid).
    """
    cx_lo = width * 0.35
    cx_hi = width * 0.65
    cy_lo = height * 0.45  # lower half = in front
    for box in track_boxes:
        x1, y1 = float(box[0]), float(box[1])
        x2, y2 = float(box[2]), float(box[3])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if cx_lo <= cx <= cx_hi and cy >= cy_lo:
            return True
    return False


# Crosswalk in front of car: distance and lateral spread (peds cross in car's path)
CROSS_DISTANCE = 10.0   # m in front of car
HALF_SPAN = 2.5         # lateral: peds at -2.5, 0, +2.5 m
# Nav search: keep peds in camera view (forward 3-15 m, lateral ±5 m; wider so Town10HD has matches)
NAV_FWD_MIN, NAV_FWD_MAX = 3.0, 15.0
NAV_RIGHT_LO, NAV_RIGHT_HI = -5.0, 5.0
MIN_PEDS_IN_VIEW = 1  # require at least this many nav points in front (spawn 1-3 peds)


def _unit_2d(v):
    """Return normalized carla.Vector3D in the xy plane (z=0) for WalkerControl direction."""
    mag = math.sqrt(v.x ** 2 + v.y ** 2)
    if mag < 1e-6:
        return carla.Vector3D(0, 0, 0)
    return carla.Vector3D(v.x / mag, v.y / mag, 0.0)


def find_spawn_with_peds_in_view(world, spawn_points, n_peds=3, n_nav_samples=250):
    """
    Pick pedestrian nav points first, then find a vehicle spawn where those peds are in view.
    Returns (spawn_index, list of up to n_peds carla.Location) or (None, []) if no spawn has enough.
    """
    nav_points = []
    for _ in range(n_nav_samples):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            nav_points.append(loc)
    if not nav_points:
        return None, []

    def in_front_of_spawn(sp_tf, loc):
        sp_loc = sp_tf.location
        fwd = sp_tf.get_forward_vector()
        right = sp_tf.get_right_vector()
        dx = loc.x - sp_loc.x
        dy = loc.y - sp_loc.y
        fwd_dist = dx * fwd.x + dy * fwd.y
        right_dist = dx * right.x + dy * right.y
        return NAV_FWD_MIN <= fwd_dist <= NAV_FWD_MAX and NAV_RIGHT_LO <= right_dist <= NAV_RIGHT_HI

    def far_enough(loc, others, min_dist=0.8):
        for o in others:
            if math.hypot(loc.x - o.x, loc.y - o.y) < min_dist:
                return False
        return True

    best_idx = None
    best_list = []
    for idx, sp_tf in enumerate(spawn_points):
        in_front = []
        for np in nav_points:
            if not in_front_of_spawn(sp_tf, np):
                continue
            if not far_enough(np, in_front):
                continue
            in_front.append(np)
            if len(in_front) >= n_peds:
                break
        if len(in_front) > len(best_list):
            best_list = in_front[:n_peds]
            best_idx = idx
        if len(best_list) >= n_peds:
            break
    return best_idx, best_list


class MultiPedestrianScenario:
    """
    Three pedestrians spawned close in front of the car (1.5 m).
    Randomness (deterministic with seed): velocity and heading angle per ped.
    """

    def __init__(self, world, vehicle, bp_lib, seed=42):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = bp_lib
        self.seed = seed
        self.walkers = []
        self.controllers = []
        self.walker_behaviors = []  # for direct control: list of {"direction": Vector3D, "speed": float}
        self._child_cross_time = None

    def _spawn_z_from_vehicle(self):
        """Use vehicle's waypoint for spawn height so Z is always valid."""
        try:
            wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
            if wp:
                return wp.transform.location.z + 0.5
        except Exception:
            pass
        return self.vehicle.get_location().z + 0.5

    def _crosswalk_ground_z(self, cross_centre):
        """Snap Z to road at crosswalk centre."""
        wp = self.world.get_map().get_waypoint(cross_centre)
        if wp:
            return wp.transform.location.z + 0.5
        return cross_centre.z + 0.5

    def _nav_spawns_in_front(self, n=3):
        """Get n nav-mesh positions in front of the car (in view). Returns list of Location."""
        found = []
        # Left, centre, right lateral bands so peds are spread across the crosswalk
        bands = [
            (NAV_RIGHT_LO, NAV_RIGHT_LO + 1.2),
            (-0.8, 0.8),
            (NAV_RIGHT_HI - 1.2, NAV_RIGHT_HI),
        ]
        for (r_lo, r_hi) in bands:
            loc = self._nav_point_in_direction(NAV_FWD_MIN, NAV_FWD_MAX, r_lo, r_hi, max_tries=80)
            if loc is not None:
                found.append(loc)
            if len(found) >= n:
                break
        # Fallback: any nav points in the in-view cone
        def _far_enough(loc, others, min_dist=0.8):
            for o in others:
                if math.hypot(loc.x - o.x, loc.y - o.y) < min_dist:
                    return False
            return True
        if len(found) < n:
            for _ in range(120):
                loc = self._nav_point_in_direction(NAV_FWD_MIN, NAV_FWD_MAX, NAV_RIGHT_LO, NAV_RIGHT_HI, max_tries=1)
                if loc is not None and _far_enough(loc, found):
                    found.append(loc)
                    if len(found) >= n:
                        break
        return found[:n]

    def _nav_point_in_direction(self, min_fwd, max_fwd, min_right, max_right, max_tries=30):
        """Return a nav-mesh point in front of the vehicle (vehicle frame: fwd, right), or None."""
        v_loc = self.vehicle.get_location()
        v_fwd = self.vehicle.get_transform().get_forward_vector()
        v_right = self.vehicle.get_transform().get_right_vector()
        for _ in range(max_tries):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            dx = loc.x - v_loc.x
            dy = loc.y - v_loc.y
            fwd_dist = dx * v_fwd.x + dy * v_fwd.y
            right_dist = dx * v_right.x + dy * v_right.y
            if min_fwd <= fwd_dist <= max_fwd and min_right <= right_dist <= max_right:
                return loc
        return None

    def _try_spawn_walker(self, bp, loc, rotation, name, fallback_locations=None, use_controller=False):
        """Spawn walker at loc; on failure try fallback_locations. Returns (walker, controller or None).
        If use_controller=False (default for direct control), no AI controller is attached."""
        z0 = loc.z if (loc.z and loc.z > 1e-5) else self._spawn_z_from_vehicle()
        loc = carla.Location(loc.x, loc.y, z0)
        walker = self.world.try_spawn_actor(bp, carla.Transform(loc, rotation))
        if walker is None and fallback_locations:
            for floc in fallback_locations:
                loc = carla.Location(floc.x, floc.y, z0)
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc, rotation))
                if walker is not None:
                    break
        if walker is None:
            print("[MULTI] WARNING: Failed to spawn %s at (%.1f, %.1f, %.1f)" % (name, loc.x, loc.y, loc.z))
            return None, None
        ctrl = None
        if use_controller:
            try:
                ctrl = self.world.spawn_actor(self.bp_lib.find('controller.ai.walker'), carla.Transform(), walker)
                self.world.tick()
            except Exception as e:
                print("[MULTI] WARNING: Failed to spawn controller for %s: %s" % (name, e))
                walker.destroy()
                return None, None
        return walker, ctrl

    def spawn_all_three(self, crosswalk_time=8.0, precomputed_nav_spawns=None):
        """
        Spawn three pedestrians with distinct behaviors (velocity/heading varied per run via seed):
        - Ped 1 (away): walks away from the car along the road.
        - Ped 2 (toward_car): on the other side of the road, walks toward the car.
        - Ped 3 (crossing): crosses the street (one side -> other side) from the start.
        Uses nav-mesh points for crosser and toward_car spawn/goal when available.
        """
        v_trans = self.vehicle.get_transform()
        v_loc = v_trans.location
        v_fwd = v_trans.get_forward_vector()
        v_right = v_trans.get_right_vector()
        vehicle_yaw = v_trans.rotation.yaw

        # Crosswalk 10 m in front of car (in the car's path); peds walk across it
        cross_centre = carla.Location(
            x=v_loc.x + v_fwd.x * CROSS_DISTANCE,
            y=v_loc.y + v_fwd.y * CROSS_DISTANCE,
            z=v_loc.z,
        )
        z_ground = self._crosswalk_ground_z(cross_centre)

        def make_loc(offset_fwd, offset_right):
            """Position on crosswalk: centre + fwd/right offsets; z snapped to road."""
            return carla.Location(
                x=cross_centre.x + v_fwd.x * offset_fwd + v_right.x * offset_right,
                y=cross_centre.y + v_fwd.y * offset_fwd + v_right.y * offset_right,
                z=z_ground,
            )

        walker_bps = list(self.bp_lib.filter('walker.pedestrian.*'))
        if not walker_bps:
            print("[MULTI] ERROR: No walker blueprints found.")
            return
        rng = random.Random(self.seed)
        child_bp = None
        for bp in walker_bps:
            if 'child' in bp.id.lower() or '02' in bp.id:
                child_bp = bp
                break
        if child_bp is None:
            child_bp = walker_bps[0]
        adult_bps = [b for b in walker_bps if b.id != child_bp.id] or walker_bps
        bps_for_peds = [
            adult_bps[rng.randint(0, len(adult_bps) - 1)] if adult_bps else walker_bps[0],
            adult_bps[rng.randint(0, len(adult_bps) - 1)] if adult_bps else walker_bps[0],
            child_bp,
        ]

        # Direct control (WalkerControl per tick) so peds reliably walk away / toward car / cross the street
        nav_spawns = list(precomputed_nav_spawns) if precomputed_nav_spawns else self._nav_spawns_in_front(3)
        v_fwd_unit = _unit_2d(v_fwd)
        v_right_unit = _unit_2d(v_right)

        cross_spawn_nav = self._nav_point_in_direction(1.0, 6.0, -4.0, -1.5, max_tries=60)
        toward_car_spawn_nav = self._nav_point_in_direction(1.5, 5.0, 3.0, 5.0, max_tries=60)

        # Three behaviors: direction (unit vector) and speed; spawn positions
        behaviors = [
            {
                "name": "away",
                "direction": v_fwd_unit,
                "spawn_fwd": 0.5, "spawn_right": 0.0,
                "yaw_base": 0.0, "yaw_range": 15.0, "speed_lo": 0.6, "speed_hi": 1.2,
                "use_nav_spawn": None,
            },
            {
                "name": "toward_car",
                "direction": carla.Vector3D(-v_fwd_unit.x, -v_fwd_unit.y, 0.0),
                "spawn_fwd": 2.0, "spawn_right": 4.5,
                "yaw_base": 180.0, "yaw_range": 15.0, "speed_lo": 0.5, "speed_hi": 1.0,
                "use_nav_spawn": toward_car_spawn_nav,
            },
            {
                "name": "crossing",
                "direction": v_right_unit,
                "spawn_fwd": 0.0, "spawn_right": -2.5,
                "yaw_base": 90.0, "yaw_range": 10.0, "speed_lo": 0.7, "speed_hi": 1.1,
                "use_nav_spawn": cross_spawn_nav,
            },
        ]
        self._child_cross_time = None
        for i in range(3):
            cfg = behaviors[i]
            if cfg.get("use_nav_spawn"):
                spawn_loc = cfg["use_nav_spawn"]
            else:
                spawn_loc = make_loc(cfg["spawn_fwd"], cfg["spawn_right"])
            yaw_off = cfg["yaw_base"] + rng.uniform(-cfg["yaw_range"], cfg["yaw_range"])
            spawn_yaw = vehicle_yaw + yaw_off
            spawn_rotation = carla.Rotation(pitch=v_trans.rotation.pitch, roll=v_trans.rotation.roll, yaw=spawn_yaw)
            speed = rng.uniform(cfg["speed_lo"], cfg["speed_hi"])
            fallbacks = [make_loc(cfg["spawn_fwd"], cfg["spawn_right"]), make_loc(0.0, 0.0)]
            if nav_spawns:
                fallbacks = list(nav_spawns) + fallbacks
            walker, ctrl = self._try_spawn_walker(
                bps_for_peds[i], spawn_loc, spawn_rotation,
                "Ped%d" % (i + 1), fallback_locations=fallbacks, use_controller=False,
            )
            if walker is None:
                print("[MULTI] WARNING: Could not spawn Ped %d (%s)." % (i + 1, cfg["name"]))
                continue
            self.walkers.append(walker)
            self.controllers.append(ctrl)
            self.walker_behaviors.append({"direction": cfg["direction"], "speed": speed})
            print("[MULTI] Ped %d: %s (direct control), speed %.2f m/s." % (i + 1, cfg["name"], speed))

        if not self.walkers:
            print("[MULTI] ERROR: No pedestrians spawned. Try --fixed-spawn, different --fixed-spawn-index or --spawn-index.")

    def update_child_route(self, sim_time, dt):
        """Ped 3: after _child_cross_time seconds, turn and cross (only when using AI controller)."""
        if len(self.controllers) < 3 or self.controllers[2] is None:
            return
        if self._child_cross_time is None or self._child_cross_time <= 0:
            return
        if sim_time < self._child_cross_time:
            return
        self._child_cross_time = -1
        c3 = self.controllers[2]
        c3.set_max_speed(0.85)
        cross_goal = getattr(self, "_child_cross_goal", None)
        if cross_goal is None:
            cross_goal = self._nav_point_in_direction(3.0, 10.0, 2.0, 8.0)
            if cross_goal is None:
                v_trans = self.vehicle.get_transform()
                v_loc = v_trans.location
                v_fwd = v_trans.get_forward_vector()
                v_right = v_trans.get_right_vector()
                cross_goal = carla.Location(
                    x=v_loc.x + v_fwd.x * 6 + v_right.x * 4,
                    y=v_loc.y + v_fwd.y * 6 + v_right.y * 4,
                    z=v_loc.z,
                )
                wp = self.world.get_map().get_waypoint(cross_goal)
                if wp:
                    cross_goal.z = wp.transform.location.z + 0.5
        c3.go_to_location(cross_goal)
        print("[MULTI] Ped 3: turning and crossing the street.")

    def cleanup(self):
        for c in self.controllers:
            if c is not None:
                try:
                    c.stop()
                except Exception:
                    pass
                try:
                    c.destroy()
                except Exception:
                    pass
        for w in self.walkers:
            try:
                w.destroy()
            except Exception:
                pass
        self.controllers = []
        self.walkers = []


def run_scenario(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if args.map and not world.get_map().name.endswith(args.map):
        world = client.load_world(args.map)
        import time
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

    # Get map spawn points
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points.")
        return

    # Pick ped nav points first, then choose a car spawn where they are in view (works even when nav is sparse)
    precomputed_ped_locs = find_spawn_with_peds_in_view(world, spawn_points, n_peds=MIN_PEDS_IN_VIEW)
    best_spawn_idx, ped_locations = precomputed_ped_locs
    if best_spawn_idx is None or not ped_locations:
        print("No spawn point has pedestrian nav points in view (4-12 m ahead). Try --map Town03.")
        return
    # Spawn vehicle at the chosen spawn (try this index first, then others if occupied)
    vehicle = None
    spawn_index = best_spawn_idx
    for i in range(len(spawn_points)):
        idx = (best_spawn_idx + i) % len(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[idx])
        if vehicle is not None:
            spawn_index = idx
            break
    if vehicle is None:
        print("No free vehicle spawn.")
        return
    world.tick()
    if args.stationary:
        vehicle.set_simulate_physics(False)
    print("[MULTI] Car at spawn index %d: %d pedestrian nav point(s) in view (4-12 m ahead)." % (spawn_index, len(ped_locations)))
    scenario = MultiPedestrianScenario(world, vehicle, bp_lib, seed=args.seed)
    use_fixed = args.fixed_spawn

    traffic_manager = None
    if not args.stationary:
        try:
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            # Drive slower: 50% below speed limit (obey traffic lights by default)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 50)
            traffic_manager.auto_lane_change(vehicle, False)
            print("[MULTI] Car using TrafficManager (obey lights, slow speed).")
        except Exception as e:
            print("[MULTI] TrafficManager not available (%s); car will use simple throttle." % e)

    # Car's forward-facing camera (driver view): look along car's path so peds crossing ahead are in frame
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(args.width))
    camera_bp.set_attribute('image_size_y', str(args.height))
    camera_bp.set_attribute('fov', '90')
    if use_fixed:
        # Lower, more downward: hood + road ahead so crossing peds are clearly in front of the car
        cam_tf = carla.Transform(
            carla.Location(x=1.2, z=1.35),
            carla.Rotation(pitch=-12.0),
        )
    else:
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-10))
    camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)

    image_queue = []
    def on_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        # BGR for OpenCV VideoWriter (same as spawn_pedestrian_video)
        image_queue.append(arr[:, :, :3][:, :, ::-1].copy())

    camera.listen(on_image)

    # Perception filter (optional): YOLO + Behavioral EKF tracker for collision prediction
    use_filter = not args.no_filter and FILTER_AVAILABLE
    if use_filter:
        _repo = Path(__file__).resolve().parent.parent
        weights = _repo / "perception" / "yolov9" / "weights" / "yolov9-c.pt"
        data = _repo / "perception" / "yolov9" / "data" / "coco128.yaml"
        mot_state = create_mot_state(weights=str(weights), data=str(data), device="cpu", imgsz=640, classes=[0])
        print("[MULTI] Using perception filter for collision prediction.")
    else:
        mot_state = None
        print("[MULTI] Using ground-truth projection for collision prediction (--no-filter or filter not available).")

    K = build_projection_matrix(args.width, args.height, 90.0)
    walker_id_by_actor = {}

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    gt_path = args.gt_out or (str(Path(args.output).with_suffix('')) + '_gt.txt')
    gt_file = open(gt_path, 'w')

    scenario.spawn_all_three(crosswalk_time=args.cross_time, precomputed_nav_spawns=ped_locations)
    for i, w in enumerate(scenario.walkers):
        walker_id_by_actor[id(w)] = i + 1

    # Let spawn settle (nav mesh, physics) so peds appear on first frames
    for _ in range(5):
        world.tick()
    while image_queue:
        image_queue.pop(0)
    print("[MULTI] Spawned %d pedestrian(s)." % len(scenario.walkers))

    sim_time = 0.0
    dt = 1.0 / args.fps
    frame_idx = 0

    try:
        if args.stationary:
            print("Starting multi-pedestrian scenario. Car is stationary; pedestrians in view.")
        else:
            print("Starting multi-pedestrian scenario. Car drives (TM); brakes on collision prediction.")
        while sim_time < args.duration:
            world.tick()
            sim_time += dt
            # Direct control: apply WalkerControl each tick so peds walk away / toward car / cross
            for w, bhv in zip(scenario.walkers, scenario.walker_behaviors):
                ctrl = carla.WalkerControl()
                ctrl.direction = bhv["direction"]
                ctrl.speed = bhv["speed"]
                ctrl.jump = False
                w.apply_control(ctrl)
            scenario.update_child_route(sim_time, dt)

            while len(image_queue) > 0:
                frame = image_queue.pop(0)
                frame_idx += 1

                # Collision prediction from filter or GT
                if use_filter and mot_state is not None:
                    tracks, _ = run_frame(mot_state, frame, draw=False)
                    track_boxes = [t[:4] for t in tracks]
                else:
                    gt_boxes = get_gt_boxes_image(
                        scenario.walkers, camera, K, args.width, args.height, walker_id_by_actor
                    )
                    track_boxes = [(left, top, left + pw, top + ph) for (_, left, top, pw, ph) in gt_boxes]

                brake = danger_zone_collision_prediction(track_boxes, args.width, args.height)
                if args.stationary:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                elif traffic_manager is not None:
                    if brake:
                        vehicle.apply_control(carla.VehicleControl(brake=1.0))
                else:
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0 if brake else 0.3, brake=1.0 if brake else 0.0, steer=0.0))

                # Record GT for evaluation
                gt_boxes = get_gt_boxes_image(
                    scenario.walkers, camera, K, args.width, args.height, walker_id_by_actor
                )
                for wid, left, top, pw, ph in gt_boxes:
                    gt_file.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1\n" % (frame_idx, wid, left, top, pw, ph))

                video_writer.write(frame)

    finally:
        gt_file.close()
        camera.stop()
        camera.destroy()
        scenario.cleanup()
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Saved video:", args.output)
        print("Saved GT:", gt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-pedestrian scenario: 3 peds, car brakes on filter prediction")
    parser.add_argument('--output', default='carla_pedestrian_crossing_multi.mp4')
    parser.add_argument('--gt-out', default='')
    parser.add_argument('--duration', type=float, default=25.0)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='', help='CARLA map (e.g. Town10HD); load if not current')
    parser.add_argument('--fixed-spawn', action='store_true',
                        help='Use a map spawn point so the car is on the road with correct orientation')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--filter', default='vehicle.tesla.model3')
    parser.add_argument('--no-filter', action='store_true', help='Use GT for collision check (no perception stack)')
    parser.add_argument('--no-stationary', action='store_true',
                        help='Car drives with TrafficManager (obey lights, slow); default is stationary')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic ped spawn (default 42)')
    parser.add_argument('--spawn-index', type=int, default=15,
                        help='Vehicle spawn point index when not using --fixed-spawn')
    parser.add_argument('--fixed-spawn-index', type=int, default=0,
                        help='Map spawn point index for --fixed-spawn (default 0; car on road, correct orientation)')
    parser.add_argument('--cross-time', type=float, default=5.0, help='Time (s) when child starts crossing')
    args = parser.parse_args()
    args.stationary = not args.no_stationary
    run_scenario(args)
