#!/usr/bin/env python3
"""
motion_planner_moveit2.py

UR5 motion planner with MoveIt2 collision checking for team_2.

Uses MoveIt2 for motion planning with collision checking, with IK fallback
when MoveIt2 planning fails.
"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint, BoundingVolume
)
from shape_msgs.msg import SolidPrimitive

from assignment_2 import move_joint as arm_api
from assignment_2 import min_jerk as traj_utils
from assignment_2 import quaternion
from assignment_1 import misc

from team_2_interfaces.srv import PlanMotion


JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def _duration_msg(seconds):
    return Duration(sec=int(seconds), nanosec=int((seconds - int(seconds)) * 1e9))


class MotionPlannerMoveIt2(Node):
    def __init__(self):
        super().__init__('motion_planner')

        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('acceleration_scaling', 0.2)
        self.declare_parameter('use_moveit', True)

        self.planning_time = float(self.get_parameter('planning_time').value)
        self.velocity_scaling = float(self.get_parameter('velocity_scaling').value)
        self.acceleration_scaling = float(self.get_parameter('acceleration_scaling').value)
        self.use_moveit = bool(self.get_parameter('use_moveit').value)

        # Reuse the working UR5 client from assignment_2/manip_challenge
        self.arm = arm_api.ArmClient()

        # Create a reentrant callback group for async ROS2 calls
        self.cb_group = ReentrantCallbackGroup()

        # MoveIt2 service client
        self.moveit_client = self.create_client(
            GetMotionPlan,
            '/plan_kinematic_path',
            callback_group=self.cb_group
        )

        # Service server
        self.srv = self.create_service(
            PlanMotion, 
            'plan_motion', 
            self.plan_motion_callback,
            callback_group=self.cb_group
        )

        # Wait for MoveIt2 service if enabled
        if self.use_moveit:
            self._wait_for_moveit()
        else:
            self.get_logger().info('[MoveIt2] Disabled by parameter')

        self.get_logger().info(
            f'MotionPlanner ready. Planning time: {self.planning_time}s, '
            f'MoveIt2: {"enabled" if self.use_moveit else "disabled"}'
        )

    def _wait_for_moveit(self):
        """Wait for MoveIt2 planning service to be available."""
        for i in range(15):
            if self.moveit_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('[MoveIt2] Planning service connected ✓')
                return True
            self.get_logger().debug(f'[MoveIt2] Waiting for service... ({i+1}/15)')
        
        self.get_logger().warn('[MoveIt2] Service not available, will use IK fallback')
        self.use_moveit = False
        return False

    def _plan_with_moveit(self, goal_pose, timeout=None):
        """
        Plan trajectory using MoveIt2 with collision checking.
        
        Returns:
            (success: bool, trajectory: JointTrajectory or None)
        """
        if not self.use_moveit or not self.moveit_client.service_is_ready():
            return False, None

        try:
            current_joints = self.arm.js_joint_position
            
            # Build MotionPlanRequest
            request = MotionPlanRequest()
            request.workspace_parameters.header.frame_id = 'world'
            request.workspace_parameters.min_corner.x = -0.5
            request.workspace_parameters.min_corner.y = -0.5
            request.workspace_parameters.min_corner.z = -0.1
            request.workspace_parameters.max_corner.x = 1.0
            request.workspace_parameters.max_corner.y = 1.0
            request.workspace_parameters.max_corner.z = 1.5

            # Start state
            request.start_state.joint_state.header.frame_id = 'world'
            request.start_state.joint_state.name = JOINT_NAMES
            request.start_state.joint_state.position = current_joints

            # Goal constraint (Cartesian pose)
            request.goal_constraints.append(Constraints())

            # Position constraint
            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = 'world'
            pos_constraint.link_name = 'tool0'
            pos_constraint.target_point = goal_pose.position
            
            # Sphere constraint around target (5cm tolerance)
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [0.05]
            
            pos_constraint.constraint_region.primitives.append(sphere)
            pos_constraint.constraint_region.primitive_poses.append(PoseStamped(
                header=__import__('std_msgs.msg').Header(frame_id='world'),
                pose=Pose(
                    position=goal_pose.position,
                    orientation=__import__('geometry_msgs.msg').Quaternion(w=1.0)
                )
            ).pose)
            pos_constraint.weight = 1.0
            request.goal_constraints[0].position_constraints.append(pos_constraint)

            # Orientation constraint
            orient_constraint = OrientationConstraint()
            orient_constraint.header.frame_id = 'world'
            orient_constraint.link_name = 'tool0'
            orient_constraint.orientation = goal_pose.orientation
            orient_constraint.absolute_x_axis_tolerance = 0.1
            orient_constraint.absolute_y_axis_tolerance = 0.1
            orient_constraint.absolute_z_axis_tolerance = 0.1
            orient_constraint.weight = 1.0
            request.goal_constraints[0].orientation_constraints.append(orient_constraint)

            request.group_name = 'ur5_arm'
            request.planner_id = 'RRTstar'
            request.allowed_planning_time = self.planning_time
            request.num_planning_attempts = 3
            request.max_velocity_scaling_factor = self.velocity_scaling
            request.max_acceleration_scaling_factor = self.acceleration_scaling

            # Send async request
            future = self.moveit_client.call_async(
                GetMotionPlan.Request(motion_plan_request=request)
            )

            # Wait for result with timeout
            if timeout is None:
                timeout = self.planning_time + 1.0
            
            start = time.time()
            while not future.done() and (time.time() - start) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)

            if not future.done():
                self.get_logger().warn('[MoveIt2] Planning timed out')
                return False, None

            result = future.result()
            error_code = result.motion_plan_response.error_code.val

            if error_code == 1:  # SUCCESS
                traj = result.motion_plan_response.trajectory.joint_trajectory
                self.get_logger().info(
                    f'[MoveIt2] Plan found ✓ ({len(traj.points)} points, '
                    f'collision-checked)'
                )
                return True, traj
            else:
                self.get_logger().warn(f'[MoveIt2] Planning failed (error code: {error_code})')
                return False, None

        except Exception as e:
            self.get_logger().error(f'[MoveIt2] Planning error: {e}')
            return False, None

    def _plan_pose_trajectory_ik(self, goal_pose, duration):
        """
        Fallback: Plan trajectory using direct IK (no collision checking).
        
        Returns:
            (time_grid, joint_positions, joint_velocities) or (None, None, None)
        """
        try:
            if self.arm.js_joint_position is None:
                raise RuntimeError('Current joint state not available')

            start_pose = self.arm.fk_request(self.arm.js_joint_position, attach_tool=True)
            time_grid, progress, _, _, _ = traj_utils.min_jerk([0], [1], duration)

            # Build interpolated pose path
            poses = []
            for p in progress[:, 0]:
                pose = Pose()
                pose.position.x = (1.0 - p) * start_pose.position.x + p * goal_pose.position.x
                pose.position.y = (1.0 - p) * start_pose.position.y + p * goal_pose.position.y
                pose.position.z = (1.0 - p) * start_pose.position.z + p * goal_pose.position.z
                pose.orientation = quaternion.slerp(
                    start_pose.orientation, goal_pose.orientation, p
                )
                poses.append(self.arm.detachTool(pose))

            q = np.array(self.arm.js_joint_position, dtype=float)
            joint_positions = [q.copy()]
            joint_velocities = [np.zeros_like(q)]

            for i in range(1, len(time_grid)):
                prev_frame = misc.pose2KDLframe(poses[i - 1])
                frame = misc.pose2KDLframe(poses[i])

                d_p = frame.p - prev_frame.p
                d_position = np.array([d_p[0], d_p[1], d_p[2]], dtype=float).reshape(3, 1)

                r_now = np.array([[frame.M[r, c] for c in range(3)] for r in range(3)])
                r_prev = np.array([[prev_frame.M[r, c] for c in range(3)] for r in range(3)])
                r_delta = r_now @ r_prev.T
                d_w = 0.5 * np.array([
                    [r_delta[2, 1] - r_delta[1, 2]],
                    [r_delta[0, 2] - r_delta[2, 0]],
                    [r_delta[1, 0] - r_delta[0, 1]],
                ])

                delta = np.vstack([d_position, d_w])
                jacobian = np.asarray(self.arm.arm_kdl.jacobian(q.tolist()))
                jacobian_inv = jacobian.T @ np.linalg.inv(
                    jacobian @ jacobian.T + 0.1**2 * np.eye(6)
                )
                dq = jacobian_inv @ delta
                q = q + dq[:, 0]

                if not np.all(np.isfinite(q)):
                    raise RuntimeError('IK produced invalid joint values')

                joint_positions.append(q.copy())
                joint_velocities.append(dq[:, 0].copy())

            return np.asarray(time_grid), np.asarray(joint_positions), np.asarray(joint_velocities)

        except Exception as e:
            self.get_logger().error(f'[IK] Planning failed: {e}')
            return None, None, None

    def _trajectory_to_msg(self, time_grid, joint_positions, joint_velocities):
        """Convert arrays to JointTrajectory message."""
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES
        for idx, t in enumerate(time_grid):
            trajectory.points.append(
                JointTrajectoryPoint(
                    positions=joint_positions[idx].tolist(),
                    velocities=joint_velocities[idx].tolist(),
                    accelerations=[0.0] * 6,
                    time_from_start=_duration_msg(float(t)),
                )
            )
        return trajectory

    def plan_motion_callback(self, request, response):
        """Service callback for motion planning."""
        self.get_logger().info('plan_motion: Received request')

        goal_pose = request.goal_pose
        self.get_logger().info(
            f'Planning to pose: x={goal_pose.position.x:.3f}, '
            f'y={goal_pose.position.y:.3f}, z={goal_pose.position.z:.3f}'
        )

        # Try MoveIt2 first (with collision checking)
        success, trajectory = self._plan_with_moveit(goal_pose)

        if not success:
            # Fallback to IK-based planning (no collision checking)
            self.get_logger().info('[Fallback] Using IK planner (no collision checking)')
            time_grid, joint_positions, joint_velocities = self._plan_pose_trajectory_ik(
                goal_pose, self.planning_time
            )

            if time_grid is None:
                response.success = False
                response.trajectory = JointTrajectory()
                return response

            trajectory = self._trajectory_to_msg(time_grid, joint_positions, joint_velocities)

        response.trajectory = trajectory
        response.success = True

        # Execute if requested
        if request.execute:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = response.trajectory
            self.get_logger().info('Executing planned trajectory via ur5_controller')
            self.arm.send_goal(goal)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerMoveIt2()
    
    # Use MultiThreadedExecutor to handle concurrent async calls
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.arm.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
