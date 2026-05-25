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
import numpy as np

from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose, Wrench
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from assignment_1 import move_joint

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']



def your_forward_kinematics(theta, return_transforms=False):
    """ Compute the end-effector pose given a set of joint angles """
    
    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    # Place your homogeneous transformation matrix here! 
    T10 = np.array([[np.cos(theta[0]), -np.sin(theta[0]), 0, 0],
           [np.sin(theta[0]), np.cos(theta[0]), 0, 0],
           [0, 0, 1, 0.089],
           [0, 0, 0, 1]])
    T21 = np.array([[np.cos(theta[1]), -np.sin(theta[1]), 0, 0],
           [0, 0, 1, 0.109],
           [-np.sin(theta[1]), -np.cos(theta[1]), 0, 0],
           [0, 0, 0, 1]])
    T32 = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0, 0.425],
           [np.sin(theta[2]), np.cos(theta[2]), 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])
    T43 = np.array([[np.cos(theta[3]), -np.sin(theta[3]), 0, 0.392],
           [np.sin(theta[3]), np.cos(theta[3]), 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])
    T54 = np.array([[np.cos(theta[4]), -np.sin(theta[4]), 0, 0],
           [0, 0, 1, 0.095],
           [-np.sin(theta[4]), -np.cos(theta[4]), 0, 0],
           [0, 0, 0, 1]])
    T65 = np.array([[np.cos(theta[5]), -np.sin(theta[5]), 0, 0],
           [0, 0, -1, -0.25],
           [np.sin(theta[5]), np.cos(theta[5]), 0, 0],
           [0, 0, 0, 1]])
    T60 = T10 @ T21 @ T32 @ T43 @ T54 @ T65
    transforms = [T10, T21, T32, T43, T54, T65]

    R = T60[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s


    # you can print out a pose message by filling followings
    # please, do not import external library like PyKDL 
    # you can import math or numpy like default libraries. 
    ps = Pose()
    ps.position.x = T60[0][3]
    ps.position.y = T60[1][3]
    ps.position.z =  T60[2][3]
    ps.orientation.w = w
    ps.orientation.x = x
    ps.orientation.y = y
    ps.orientation.z = z
    #------------------------------------------------------------
    
    if return_transforms:
        return ps, transforms
    else:
        return ps

def send_command(node, theta, duration=3):
    
    # construct a goal message
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    g.trajectory.points.append(
        JointTrajectoryPoint(positions=theta, velocities=[0]*6,
                                 time_from_start=Duration(sec=int(duration),
            nanosec=int((duration-int(duration))*1e9)))
        )    
    move_joint.move_joint(node, g)


def main(args=None):
    
    rclpy.init() 
    
    node = rclpy.create_node("arm_client")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("init_ur5: direct control mode")
    
    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    # Place your desired joint angles! 
    
    # Problem 1.C (i)
    theta = [0,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0]
    node.get_logger().info("{}".format(your_forward_kinematics(theta)))
    send_command(node, theta)

    # stop for a while
    rclpy.spin_once(node, timeout_sec=3)

    # Problem 1.C (ii)
    theta = [-np.pi/4,-np.pi/4,np.pi/2,0,0,0]
    node.get_logger().info("{}".format(your_forward_kinematics(theta)))
    send_command(node, theta)
    #------------------------------------------------------------
    
    node.destroy_node()
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()
        


