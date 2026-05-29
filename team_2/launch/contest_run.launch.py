#!/usr/bin/env python3
"""
contest_run.launch.py

Brings up the team_2 system in STANDBY:
  - instruction_parser : /task_commands -> Qwen      -> /parsed_tasks
  - perception_node    : serves 'detect_objects' (Grounding DINO, 2D boxes)
  - grasp_server       : serves 'lift_bbox_to_3d'  (bbox -> 3D pose; depth/cloud)
  - orchestrator_node  : owns the arm; consumes /parsed_tasks and drives the
                         scout -> detect -> lift -> grasp -> place loop

Nothing moves until a command is published on /task_commands, which the parser
turns into /parsed_tasks; the orchestrator's latched subscription is the trigger
that leaves the competition's required "standby state".

Run:
  ros2 launch team_2 contest_run.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('team_2')
    params = os.path.join(pkg_share, 'config', 'params.yaml')

    common = dict(package='team_2', output='screen',
                  emulate_tty=True, parameters=[params])

    return LaunchDescription([
        Node(executable='instruction_parser',
             name='instruction_parser', **common),
        Node(executable='perception_node',
             name='perception_node', **common),
        # SEAM 5: the bbox -> 3D lift service (refactor of example5_grasp_server).
        Node(executable='grasp_server',
             name='grasp_server', **common),
        Node(executable='orchestrator_node',
             name='orchestrator_node', **common),
    ])