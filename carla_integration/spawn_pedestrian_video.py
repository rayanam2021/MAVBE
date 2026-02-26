#!/usr/bin/env python
"""
CARLA script: spawn a vehicle, spawn a pedestrian every 10 seconds with
random walking behavior, run for 60 seconds, and save the camera view as video.

Requirements: CARLA server running (default 127.0.0.1:2000), and CARLA PythonAPI
on PYTHONPATH or in ../carla, or carla egg in ../carla/dist/.

Usage:
    python spawn_pedestrian_video.py [--host 127.0.0.1] [--port 2000] [--output out.mp4] [--fps 20]
"""

from __future__ import print_function

import argparse
import glob
import os
import random
import sys
import time

try:
    import numpy as np
    import cv2
except ImportError as e:
    print("Need numpy and opencv-python: pip install numpy opencv-python")
    sys.exit(1)

# Add CARLA PythonAPI to path (same as trajectory_planning.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MAVBE_ROOT = os.path.dirname(_SCRIPT_DIR)
try:
    sys.path.append(glob.glob(os.path.join(_MAVBE_ROOT, 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append(os.path.join(_MAVBE_ROOT, 'carla'))

try:
    import carla
except ImportError:
    print("CARLA module not found. Add CARLA PythonAPI to PYTHONPATH or place it in MAVBE/carla/")
    sys.exit(1)


def get_latest_image(image_holder):
    """Return latest BGR array from camera callback storage."""
    if not image_holder["data"]:
        return None
    return image_holder["data"]


def run(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Prefer Town10HD if available (paper uses it); otherwise default map
    if args.map and world.get_map().name != args.map:
        world = client.load_world(args.map)

    # Synchronous mode so we can tick and get deterministic frames
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No vehicle spawn points in this map.")
        return

    # Spawn vehicle (ego)
    vehicle_bp = blueprint_library.filter(args.filter)[0]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Failed to spawn vehicle.")
        return
    print("Spawned vehicle at", spawn_point.location)

    # RGB camera attached to vehicle (hood/dashboard view)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(args.width))
    camera_bp.set_attribute('image_size_y', str(args.height))
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.0),
        carla.Rotation(pitch=-15.0)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Holder for latest camera frame (callback writes, main loop reads)
    image_holder = {"data": None}

    def on_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_holder["data"] = array[:, :, :3][:, :, ::-1].copy()  # BGRA -> BGR

    camera.listen(on_image)

    # Video writer
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

    walkers = []
    controllers = []
    next_pedestrian_time = 10.0  # first pedestrian at t=10s
    last_redirect_time = 0.0     # for random walk redirect every 5s
    duration = 60.0
    sim_time = 0.0
    dt = 1.0 / args.fps

    try:
        while sim_time < duration:
            world.tick()
            sim_time += dt

            # Spawn a new pedestrian every 10 seconds
            if sim_time >= next_pedestrian_time:
                next_pedestrian_time += 10.0
                try:
                    # Spawn near the vehicle (offset in vehicle frame)
                    v_t = vehicle.get_transform()
                    # Offset 5–12 m in front/side
                    dx = random.uniform(5, 12) * (1 if random.random() > 0.5 else -1)
                    dy = random.uniform(5, 12) * (1 if random.random() > 0.5 else -1)
                    loc = v_t.location
                    spawn_loc = carla.Location(
                        x=loc.x + dx,
                        y=loc.y + dy,
                        z=loc.z + 0.5
                    )
                    try:
                        waypoint = world.get_map().get_waypoint(vehicle.get_location())
                        if waypoint:
                            spawn_loc.z = waypoint.transform.location.z + 0.5
                    except Exception:
                        pass
                    spawn_transform = carla.Transform(spawn_loc, carla.Rotation(yaw=random.uniform(0, 360)))

                    walker_bps = blueprint_library.filter('walker.pedestrian.*')
                    if not walker_bps:
                        walker_bps = blueprint_library.filter('walker.*')
                    if not walker_bps:
                        print("No walker blueprints in this build, skipping pedestrian.")
                        continue
                    walker_bp = random.choice(walker_bps)
                    walker = world.try_spawn_actor(walker_bp, spawn_transform)
                    if walker is None:
                        continue
                    walkers.append(walker)

                    controller_bp = blueprint_library.find('controller.ai.walker')
                    if controller_bp is None:
                        print("No controller.ai.walker in this build, pedestrian will not move.")
                        continue
                    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                    controllers.append(controller)
                    world.tick()
                    controller.start()
                    controller.set_max_speed(1.0 + random.uniform(0, 0.6))
                    controller.go_to_location(world.get_random_location_from_navigation())
                    print("Spawned pedestrian at t={:.1f}s".format(sim_time))
                except Exception as e:
                    print("Pedestrian spawn failed:", e)

            # Give existing pedestrians new random destinations every 5 seconds (random behavior)
            if sim_time - last_redirect_time >= 5.0:
                last_redirect_time = sim_time
                for c in controllers:
                    try:
                        dest = world.get_random_location_from_navigation()
                        if dest is not None:
                            c.go_to_location(dest)
                    except Exception:
                        pass

            frame = get_latest_image(image_holder)
            if frame is not None:
                video_writer.write(frame)
    finally:
        # Cleanup
        camera.stop()
        camera.destroy()
        for c in controllers:
            try:
                c.stop()
                c.destroy()
            except Exception:
                pass
        for w in walkers:
            try:
                w.destroy()
            except Exception:
                pass
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Saved video to", os.path.abspath(args.output))
        print("Spawned", len(walkers), "pedestrians in 60 seconds.")


def main():
    parser = argparse.ArgumentParser(
        description="Spawn vehicle and pedestrians in CARLA every 10s for 60s, save video."
    )
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--output', default='carla_pedestrian_60s.mp4', help='Output video path')
    parser.add_argument('--fps', type=float, default=20.0, help='Simulation and video FPS')
    parser.add_argument('--width', type=int, default=1920, help='Camera width')
    parser.add_argument('--height', type=int, default=1080, help='Camera height')
    parser.add_argument('--map', default='', help='Map name (e.g. Town10HD); leave empty for current')
    parser.add_argument('--filter', default='vehicle.tesla.model3',
                        help='Vehicle blueprint filter (default: vehicle.tesla.model3)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
