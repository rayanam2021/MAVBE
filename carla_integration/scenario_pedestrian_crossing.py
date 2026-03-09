#!/usr/bin/env python
"""
CARLA script: Create controlled pedestrian crossing scenarios.
Spawns an ego-vehicle and pedestrians at specific relative locations
with deterministic trajectories for MOT testing.

Usage:
    python scenario_pedestrian_crossing.py [--output out.mp4] [--duration 15.0]
"""

import argparse
import os
import random
import sys
import glob
import time
from pathlib import Path

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
# Look in the user's specific CARLA installation path
carla_root = os.path.expanduser("~/autonomy_projects/CARLA_0.9.16")
try:
    sys.path.append(glob.glob(os.path.join(carla_root, 'PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

# Also add the agents directory for navigation tools
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))

import cv2
import numpy as np

try:
    import carla
except ImportError:
    print("CARLA module not found.")
    print(f"Looked in: {carla_root}")
    print("Please ensure your 'carla_env' is active or PYTHONPATH is set correctly.")
    sys.exit(1)

class PedestrianScenario:
    """Helper class to manage relative spawning and trajectories."""
    def __init__(self, world, vehicle, blueprint_library):
        self.world = world
        self.vehicle = vehicle
        self.bp_lib = blueprint_library
        self.walkers = []
        self.controllers = []

    def spawn_crossing_pedestrian(self, forward_offset=10.0, side_offset=5.0, target_side_offset=-5.0, speed=1.5):
        """
        Spawns a pedestrian at a relative offset and commands them to cross the road.
        """
        v_trans = self.vehicle.get_transform()
        v_fwd = v_trans.get_forward_vector()
        v_right = v_trans.get_right_vector()

        # Calculate spawn location (Relative to vehicle transform)
        spawn_loc = v_trans.location + (v_fwd * forward_offset) + (v_right * side_offset)
        spawn_loc.z += 0.5 
        
        # Snap to navigation mesh
        waypoint = self.world.get_map().get_waypoint(spawn_loc)
        if waypoint:
            spawn_loc.z = waypoint.transform.location.z + 0.5

        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        walker_bp = random.choice(walker_bps)
        
        spawn_trans = carla.Transform(spawn_loc, carla.Rotation(yaw=v_trans.rotation.yaw - 90))
        walker = self.world.try_spawn_actor(walker_bp, spawn_trans)
        
        if walker:
            self.walkers.append(walker)
            controller_bp = self.bp_lib.find('controller.ai.walker')
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
            self.controllers.append(controller)
            
            controller.start()
            target_loc = v_trans.location + (v_fwd * (forward_offset + 2.0)) + (v_right * target_side_offset)
            controller.go_to_location(target_loc)
            controller.set_max_speed(speed)
            
            print(f"[SCENARIO] Spawned pedestrian: {forward_offset}m ahead, {side_offset}m -> {target_side_offset}m side.")
            return walker
        return None

    def cleanup(self):
        for c in self.controllers:
            try:
                c.stop()
                c.destroy()
            except: pass
        for w in self.walkers:
            try:
                w.destroy()
            except: pass
        self.controllers = []
        self.walkers = []

def run_scenario(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    
    vehicle_bp = bp_lib.filter(args.filter)[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(args.width))
    camera_bp.set_attribute('image_size_y', str(args.height))
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-10))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_queue = []
    def on_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_queue.append(array[:, :, :3])

    camera.listen(on_image)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

    scenario_mgr = PedestrianScenario(world, vehicle, bp_lib)
    sim_time = 0.0
    dt = 1.0 / args.fps

    try:
        print(f"Starting Scenario in {world.get_map().name}...")
        while sim_time < args.duration:
            world.tick()
            sim_time += dt

            # Events
            if 2.0 <= sim_time < 2.0 + dt:
                scenario_mgr.spawn_crossing_pedestrian(forward_offset=12.0, side_offset=6.0, target_side_offset=-6.0, speed=1.6)
            if 5.0 <= sim_time < 5.0 + dt:
                scenario_mgr.spawn_crossing_pedestrian(forward_offset=18.0, side_offset=-5.0, target_side_offset=8.0, speed=1.2)
            if 8.0 <= sim_time < 8.0 + dt:
                scenario_mgr.spawn_crossing_pedestrian(forward_offset=10.0, side_offset=7.0, target_side_offset=-7.0, speed=2.5)

            while len(image_queue) > 0:
                frame = image_queue.pop(0)
                video_writer.write(frame)

    finally:
        print("Cleaning up...")
        camera.stop()
        camera.destroy()
        scenario_mgr.cleanup()
        vehicle.destroy()
        video_writer.release()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"Saved video to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Controlled CARLA Pedestrian Scenarios")
    parser.add_argument('--output', default='carla_pedestrian_crossing.mp4')
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--filter', default='vehicle.tesla.model3')
    run_scenario(parser.parse_args())
