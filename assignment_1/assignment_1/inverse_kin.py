#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import rclpy
import rclpy.node

from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseArray, Pose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
from assignment_1 import move_joint, forward_kin

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def move_position(node, goal_pose, init_joint=None):
    """ 
    A function to send a list of joint angles to the robot. 
    This function waits for completing the commanded motion.

    Parameters
    ----------
    node:  
        a ROS2 node handle
    goal_pose : Pose
        a Pose message from geometry_msgs
    init_joint : list
        a initial/current angle
    """

    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    # 1) construct forward kinematics (problem 1)
    forward_kinematics = forward_kin.your_forward_kinematics
    
    # 2) Get the start position from the init_joint via forward kinematics
    start_pose = forward_kinematics(init_joint)
    start_position = [start_pose.position.x, start_pose.position.y, start_pose.position.z]
    # print(f"Start position: {start_position}")
    # 3) Get the goal position from the goal_pose
    goal_position = [goal_pose.position.x, goal_pose.position.y, goal_pose.position.z]
    # print(f"Goal position: {goal_position}")

    # 4) Get a pose/position trajectory from start to goal positions
    pose_traj = np.linspace(start_position, goal_position, 101)

    
    # construct a goal message
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
        
    # Get a sequence of joint angles that track the position trajecotory
    q = np.array(init_joint, dtype=float)
    print(f"At 0% of the trajectory, q = {q}")
    dt = 0.05
    _lambda = 0.1
    time_from_start = 0
    for i in range(1, len(pose_traj)):
        # Recompute forward kinematics at current q
        _, transforms = forward_kinematics(q, return_transforms=True)

        # Find a Jacobian
        T10 = transforms[0]
        T20 = T10 @ transforms[1]
        T30 = T20 @ transforms[2]
        T40 = T30 @ transforms[3]
        T50 = T40 @ transforms[4]
        T60 = T50 @ transforms[5]

        a1_local = np.array([0, 0, 1])
        a2_local = np.array([0, 1, 0])
        a3_local = np.array([0, 0, 1])
        a4_local = np.array([0, 0, 1])
        a5_local = np.array([0, 1, 0])
        a6_local = np.array([0,-1, 0])

        z0 = a1_local                       # joint 1 axis in base
        z1 = T10[:3,:3] @ a2_local          # joint 2 axis in base
        z2 = T20[:3,:3] @ a3_local          # joint 3
        z3 = T30[:3,:3] @ a4_local          # joint 4
        z4 = T40[:3,:3] @ a5_local          # joint 5
        z5 = T50[:3,:3] @ a6_local          # joint 6

        o0 = np.array([0, 0, 0])       # base frame origin
        o1 = T10[:3, 3]
        o2 = T20[:3, 3]
        o3 = T30[:3, 3]
        o4 = T40[:3, 3]
        o5 = T50[:3, 3]
        o6 = T60[:3, 3]                # end-effector

        J_p = np.array([
            np.cross(z0, o6 - o0),
            np.cross(z1, o6 - o1),
            np.cross(z2, o6 - o3),   
            np.cross(z3, o6 - o4),
            np.cross(z4, o6 - o4),
            np.cross(z5, o6 - o5),
        ]).T
        J_o = np.array([z0, z1, z2, z3, z4, z5]).T
        J = np.vstack((J_p, J_o))
        J_inv = J_p.T @ np.linalg.inv(J_p @ J_p.T + _lambda**2 * np.eye(3)) 

        if i == 1:
            print(f"J_p: {J_p}")
            print(f"psudo inverse of J: {J_inv}")

        # Compute a delta position
        dx = pose_traj[i] - pose_traj[i-1]

        # Compute a delta theta
        dtheta = J_inv @ dx

        # Compute a desired theta
        q += dtheta

        if i in [25, 50, 75, 100]:
             print(f"At {i}% of the trajectory, q = {q}")

        time_from_start += dt
        g.trajectory.points.append(
           JointTrajectoryPoint(positions=q, velocities=[0]*6,
                                time_from_start=Duration(sec=int(time_from_start),
           nanosec=int((time_from_start-int(time_from_start))*1e9)))
           )
    #------------------------------------------------------------
        
    move_joint.move_joint(node, g)


def main(args=None):
    
    rclpy.init() 
    
    node = rclpy.create_node("arm_client")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("init_ur5: direct control mode")

    # Problem 2
    # Move to the initial joint configuration
    theta    = [-0.8, -1.5708, 1.5708, -1.5708, -1.5708, -0.8]
    duration = 4
    
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    g.trajectory.points.append(
        JointTrajectoryPoint(positions=theta, velocities=[0]*6,
                                 time_from_start=Duration(sec=int(duration),
            nanosec=int((duration-int(duration))*1e9)))
        )        
    move_joint.move_joint(node, g)

    # Move following the linear position trajectory
    goal = Pose()
    goal.position.x = 0.418
    goal.position.y = 0.273
    goal.position.z = 0.264
    move_position(node, goal, theta)
    
    node.destroy_node()
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()
        


