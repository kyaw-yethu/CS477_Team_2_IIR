#!/usr/bin/python3
"""
ur5_moveit_simple.launch.py

Simplified MoveIt + Gazebo launch for team_2.
Non-interactive, works out of the box.

Usage:
  ros2 launch ur5_ros2_moveit2 ur5_moveit_simple.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            import yaml
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    use_sim_time = True
    
    # UR5 description
    ur5_description_path = get_package_share_directory('ur5_ros2_gazebo')
    xacro_file = os.path.join(ur5_description_path, 'urdf', 'ur5.urdf.xacro')
    
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc, mappings={
        "cell_layout_1": "true",
        "cell_layout_2": "false",
        'hardware_interface': "PositionJointInterface",
        'camera_enabled': "false",
        "EE_no": "true",
    })
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}
    
    # SRDF (semantic description)
    robot_description_semantic_config = load_file("ur5_ros2_moveit2", "config/ur5.srdf")
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_config}
    
    # Kinematics
    kinematics_yaml = load_yaml("ur5_ros2_moveit2", "config/kinematics.yaml")
    
    # OMPL planning
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_yaml = load_yaml("ur5_ros2_moveit2", "config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)
    
    # Controllers
    moveit_simple_controllers_yaml = load_yaml("ur5_ros2_moveit2", "config/ur5_controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    
    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }
    
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    
    # Gazebo
    gazebo_pkg = get_package_share_directory('ur5_ros2_gazebo')
    gazebo_launch = os.path.join(gazebo_pkg, 'launch', 'ur5_simulation.launch.py')
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={
            "cell_layout_1": "true",
            "cell_layout_2": "false",
            "hardware_interface": "PositionJointInterface",
            "camera_enabled": "false",
        }.items(),
    )
    
    # Spawn robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'ur5'],
        output='screen',
    )
    
    # MoveGroup node
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )
    
    # RViz
    rviz_base = get_package_share_directory("ur5_ros2_moveit2")
    rviz_config = os.path.join(rviz_base, "config", "ur5_moveit2.rviz")
    
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            {"use_sim_time": use_sim_time},
        ],
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    
    # Nodes to include
    nodes = [
        gazebo,
        spawn_entity,
        joint_state_publisher,
        run_move_group_node,
        rviz,
    ]
    
    return LaunchDescription(nodes)
