#!/usr/bin/python3
"""
ur5_moveit_external.launch.py

Launch MoveIt move_group for the UR5 without spawning Gazebo.
Use this after launching the existing manip_challenge scene.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
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
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    rviz = LaunchConfiguration('rviz', default='false')

    # Robot description from ur5_ros2_gazebo xacro
    ur5_description_path = get_package_share_directory('ur5_ros2_gazebo')
    xacro_file = os.path.join(ur5_description_path, 'urdf', 'ur5.urdf.xacro')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc, mappings={
        'cell_layout_1': 'true',
        'cell_layout_2': 'false',
        'hardware_interface': 'PositionJointInterface',
        'camera_enabled': 'false',
        'EE_no': 'true',
    })
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}

    # Semantic description + kinematics
    robot_description_semantic_config = load_file('ur5_ros2_moveit2', 'config/ur5.srdf')
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_config}
    kinematics_yaml = load_yaml('ur5_ros2_moveit2', 'config/kinematics.yaml')

    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_yaml = load_yaml('ur5_ros2_moveit2', 'config/ompl_planning.yaml')
    if ompl_planning_yaml is not None:
        ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

    moveit_simple_controllers_yaml = load_yaml('ur5_ros2_moveit2', 'config/ur5_controllers.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    trajectory_execution = {
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {'use_sim_time': use_sim_time},
        ],
    )

    rviz_full_config = os.path.join(get_package_share_directory('ur5_ros2_moveit2'), 'config', 'ur5_moveit2.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config],
        parameters=[robot_description, robot_description_semantic, kinematics_yaml, {'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock'),
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RViz?'),
        run_move_group_node,
        rviz_node,
    ])
