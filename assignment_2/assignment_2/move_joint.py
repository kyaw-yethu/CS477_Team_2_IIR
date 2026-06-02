#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys, time
import numpy as np, math
import threading
import argparse

import rclpy
import rclpy.node
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from rosidl_runtime_py import message_to_yaml
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
import xacro

from control_msgs.msg import *
from trajectory_msgs.msg import *
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseArray, Pose

import PyKDL
from pykdl_utils.kdl_kinematics import create_kdl_kin
from hrl_geom.pose_converter import PoseConv
from assignment_2 import quaternion
from assignment_1 import misc

from assignment_2 import min_jerk as mj

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class ArmClient(Node):

    def __init__(self, freq=100):
        """A function to initialize parameters and comm. functions."""        
        super().__init__('arm_client')
        
        self.freq = freq #Hz
        
        self.js_lock = threading.Lock()
        self.js_joint_name     = None
        self.js_joint_position = None

        # Contruct forward kinematics
        # Since the robot URDF does not have the user-defined gripper_link,
        # we manually attach the transformation from the robotiq_85_base_link
        # to gripper_link using self.tool_offset_frame.
        ur5_description_path = os.path.join(
            get_package_share_directory('ur5_ros2_gazebo'))
        xacro_file = os.path.join(ur5_description_path,
                                      'urdf',
                                      'ur5.urdf.xacro')
        doc = xacro.parse(open(xacro_file))
        xacro.process_doc(doc, mappings={
            "cell_layout_1": 'false',
            "cell_layout_2": 'true',
            "hardware_interface": 'PositionJointInterface',})
        robot_description = doc.toxml()
        
        self.arm_kdl = create_kdl_kin("base_link", "robotiq_85_base_link",
                                          urdf_xml=robot_description)
        self.tool_offset_frame = misc.list2KDLframe([0.15, 0.003, 0,
                                                         1.5707, 0, 1.5707])
        
        self.initComms()
        self.get_logger().info("Initialized UR5")
        
        
    def initComms(self):
        """A function to initialize communication functions."""

        self._callback_group = ReentrantCallbackGroup() 
        
        # Controller
        self.client =\
          ActionClient(self, FollowJointTrajectory,
                           '/ur5_controller/follow_joint_trajectory',
                           callback_group=self._callback_group)
        self.get_logger().info( "Waiting for ur5_arm server..." )
        server_up = self.client.wait_for_server(5)

        if not server_up:
            self.get_logger().error("Timed out waiting for Joint Trajectory"
                                    " Action Server to connect. Start the action server"
                                    " before running example.")
            self.destroy_node()
            rclpy.shutdown()
            sys.exit()
        self.get_logger().info("Connected to ur5_arm server")

        # Subscriber
        self.create_subscription(JointTrajectoryControllerState,
                                     '/ur5_controller/state',
                                     self.state_callback,
                                     10)
        while rclpy.ok():
            if self.js_joint_name is not None and self.js_joint_position is not None: break
            rclpy.spin_once(self, timeout_sec=0)


    def state_callback(self, msg):
        """ 
        A callback function that subscribes joint states.

        Parameters
        ----------
        msg : control_msgs/JointTrajectoryControllerState
            a message of desired, actual, and error state of joints
        """
        ## self.js_lock.acquire()
        self.js_joint_name     = msg.joint_names
        self.js_joint_position = list(msg.desired.positions)

        # if there is no desired position information, use the actual position
        if len(self.js_joint_position)<1:
            self.js_joint_position = list(msg.actual.positions)
        ## self.js_lock.release()

    def pose_to_joints(self, pose, q_seed=None, attach_tool=True,
                    n_interp=60, refine_iters=150,
                    tol_pos=2e-3, tol_rot=2e-2, damp=0.05):
        if q_seed is None:
            q_seed = self.js_joint_position
        if q_seed is None:
            self.get_logger().error('pose_to_joints: no joint seed.'); return None
        if attach_tool:
            pose = self.detachTool(pose)

        gf = misc.pose2KDLframe(pose)
        p_goal = np.array([gf.p[0], gf.p[1], gf.p[2]])
        R_goal = np.array([[gf.M[i, j] for j in range(3)] for i in range(3)])

        def fk(q):
            T = np.asarray(self.arm_kdl.forward(q)); return T[:3, :3], T[:3, 3]
        def ang_err(Rd, Rc):
            E = Rd @ Rc.T
            return 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])
        def so3_log(R):
            c = np.clip((np.trace(R)-1.0)/2.0, -1.0, 1.0); th = np.arccos(c)
            if th < 1e-9: return np.zeros(3)
            return th/(2.0*np.sin(th))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        def so3_exp(w):
            th = np.linalg.norm(w)
            if th < 1e-9: return np.eye(3)
            k = w/th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
            return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
        def step(q, ep, ew):
            e = np.concatenate([ep, ew]).reshape(6, 1)
            J = np.asarray(self.arm_kdl.jacobian(q))
            return q + (J.T @ np.linalg.inv(J@J.T + damp**2*np.eye(6)) @ e)[:, 0]

        q = np.array(q_seed, dtype=float)
        R0, p0 = fk(q)
        dR = so3_log(R0.T @ R_goal)
        for i in range(1, n_interp+1):
            s = i/float(n_interp)
            Rc, pc = fk(q)
            q = step(q, ((1-s)*p0 + s*p_goal) - pc, ang_err(R0 @ so3_exp(s*dR), Rc))

        best_q, best = q.copy(), 1e9
        for _ in range(refine_iters):
            Rc, pc = fk(q)
            ep, ew = p_goal - pc, ang_err(R_goal, Rc)
            c = np.linalg.norm(ep) + np.linalg.norm(ew)
            if c < best: best, best_q = c, q.copy()
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(ew) < tol_rot:
                self.get_logger().warn(
                    f'IK CONVERGED |ep|={np.linalg.norm(ep)*1000:.1f}mm '
                    f'|ew|={np.degrees(np.linalg.norm(ew)):.2f}deg')
                return [float(v) for v in q]
            q = step(q, ep, ew)

        Rc, pc = fk(best_q)
        self.get_logger().warn(
            f'IK STUCK best |ep|={np.linalg.norm(p_goal-pc)*1000:.1f}mm '
            f'|ew|={np.degrees(np.linalg.norm(ang_err(R_goal, Rc))):.2f}deg -> None, caller falls back')
        return None

    
    def fk_request(self, joints, attach_tool=True):
        """ 
        A function to compute forward kinematics.

        Parameters
        ----------
        joints : list
            joint angles
        attach_tool : Boolean
            enable to attach offset to the end effector

        Returns
        -------
        pose : geometry_msgs/Pose
            a Pose object (after attaching the tool offset)
        """
        homo_mat = self.arm_kdl.forward(joints)
        pos, quat = PoseConv.to_pos_quat(homo_mat)
        pose = misc.list2Pose(list(pos)+list(quat))
        if attach_tool: return self.attachTool(pose)
        return pose
        

    def attachTool(self, pose):
        """ 
        A function to attach tool offset on the given pose.

        Parameters
        ----------
        pose : geometry_msgs/Pose
            a Pose object

        Returns
        -------
        pose : geometry_msgs/Pose
            a Pose object after attaching the tool offset
        """
        tool_frame = misc.pose2KDLframe(pose) * self.tool_offset_frame
        return misc.KDLframe2Pose(tool_frame)
        
    def detachTool(self, pose):
        """ 
        A function to detach tool offset from the given pose.

        Parameters
        ----------
        pose : geometry_msgs/Pose
            a Pose object

        Returns
        -------
        pose : geometry_msgs/Pose
            a Pose object after detaching the tool offset
        """
        ee_frame = misc.pose2KDLframe(pose) * self.tool_offset_frame.Inverse()
        return misc.KDLframe2Pose(ee_frame)

    
    def send_goal(self, goal):
        """ 
        A function that sends a trajectory goal message.

        Parameters
        ----------
        goal : FollowJointTrajectory.Goal()
            a goal message
        """

        # Send the goal message
        goal_future = self.client.send_goal_async(goal)
        # Return result when the command is delivered
        rclpy.spin_until_future_complete(self, goal_future)

        goal_handle = goal_future.result()
        self.get_logger().info('goal_handle:\n {}'.format(goal_handle))
    
        # Wait until the execution ends and return the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
    
        result = result_future.result()
        if result is None:
            raise RuntimeError(
                'Exception while getting result: {!r}'.format(result_future.exception()))
        self.get_logger().info('Result:\n    {}'.format(message_to_yaml(result.result)))
        
        
    def move_joint(self, angles, duration=5.):
        """ 
        A function that sends a trajectory of joint angles
        to reach the desired angles.

        Parameters
        ----------
        angles : list
            a list of desired angles
        duration : float
            time duration
        """
        # Generate a min_jerk trajectory
        time, pos, vel, acc, jerk, = mj.min_jerk(self.js_joint_position, angles, duration)

        # Create a goal message
        g                        = FollowJointTrajectory.Goal()
        g.trajectory             = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES
        g.trajectory.points      = []

        # Set the trajectory to the goal message
        # ------------------------------------------------------
        # Place your code here
        # ------------------------------------------------------
        for i, t in enumerate(time): 
           g.trajectory.points.append(
               JointTrajectoryPoint(positions= pos[i],
                                    velocities= vel[i],
                                    accelerations= acc[i],
                                    time_from_start=Duration(sec=int(t), nanosec=int((t-int(t))*1e9) )))

        # ------------------------------------------------------

        self.send_goal(g)
        return time, pos, vel, acc, jerk
        
        
    def move_position(self, pose, duration=5.):
        """ 
        A function that moves the end effector to a target position.

        Parameters
        ----------
        pose : Pose
            a target pose of the end effector
        duration : float
            time duration
        """        
        # Get a goal pose
        goal_pose  = pose
        
        # Get a start pose 
        start_pose = self.fk_request(self.js_joint_position, attach_tool=True)
        # print(f"start_pose: {start_pose}")
        # print(f"goal_pose: {goal_pose}")
        # Get a sequence of path variables from the min jerk trajectory planning
        time, progress, _, _, _, = mj.min_jerk([0], [1], duration)

        # get a trajectory by interpolating from the start and the goal
        poses = []
        for i, p in enumerate(progress[:,0]):
            pose = Pose()
            # ------------------------------------------------------
            # Place your code here
            # ------------------------------------------------------
            # position
            pose.position.x = (1-p) * start_pose.position.x + p * goal_pose.position.x
            pose.position.y = (1-p) * start_pose.position.y + p * goal_pose.position.y
            pose.position.z = (1-p) * start_pose.position.z + p * goal_pose.position.z
            # ------------------------------------------------------
            poses.append(pose)


        g                        = FollowJointTrajectory.Goal()
        g.trajectory             = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES        
        q                        = self.js_joint_position        
        time_from_start          = 0

        joint_position_traj = [list(q)]
        joint_velocity_traj = [[0.]*len(q)]
        
        for i, t in enumerate(time):
            if i==0: continue
                            
            # ------------------------------------------------------
            # Place your code here
            # ------------------------------------------------------
            # get delta position
            dx = poses[i].position.x - poses[i-1].position.x
            dy = poses[i].position.y - poses[i-1].position.y
            dz = poses[i].position.z - poses[i-1].position.z
            delta_pos = np.array([dx, dy, dz]).reshape(3,1)

            # get a jacobian
            J = np.asarray(self.arm_kdl.jacobian(q))
            Jp = J[:3]

            # get a pseudo inverse of jacobian mtx
            J_inv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 0.1**2 * np.eye(3)) 
            

            # get the joint angles via IK
            dq = J_inv @ delta_pos
            q = q + dq[:, 0]
            
            # ------------------------------------------------------
            
            g.trajectory.points.append(
                JointTrajectoryPoint(positions=q,
                                         velocities=[0]*6,
                                         time_from_start=Duration(sec=int(t),
                                                                      nanosec=int((t-int(t))*1e9) )
                                         )
                )

            joint_position_traj.append(q.tolist())
            joint_velocity_traj.append(np.array(dq)[:,0].tolist())
            
            
        self.send_goal(g)

        return time, np.array(joint_position_traj), np.array(joint_velocity_traj), None, None


    def move_pose(self, pose, duration=5.):
        """ 
        A function that moves the end effector to a target pose (position+orientation).

        Parameters
        ----------
        pose : Pose
            a target pose of the end effector
        duration : float
            time duration
        """        
        # Get a goal pose
        goal_pose  = pose
        
        # Get a start pose 
        start_pose = self.fk_request(self.js_joint_position, attach_tool=True)

        # print(f"start_pose: {start_pose}")
        # print(f"goal_pose: {goal_pose}")

        # Get a sequence of path variables from the min jerk trajectory planning
        time, progress, _, _, _, = mj.min_jerk([0], [1], duration)

        # get a trajectory by interpolating from the start and the goal
        poses = []
        for i, p in enumerate(progress[:,0]):
            pose = Pose()
            # ------------------------------------------------------
            # Place your code here
            # ------------------------------------------------------
            # position (copy from move_position)
            pose.position.x = (1-p) * start_pose.position.x + p * goal_pose.position.x
            pose.position.y = (1-p) * start_pose.position.y + p * goal_pose.position.y
            pose.position.z = (1-p) * start_pose.position.z + p * goal_pose.position.z
            
            # orientation (use the SLERP function in the quaternion.py)
            pose.orientation = quaternion.slerp(start_pose.orientation, goal_pose.orientation, p)
            
            # ------------------------------------------------------
            
            # detach the tool before applying inverse kinematics (if needed)
            pose = self.detachTool(pose)
            poses.append(pose)


        g = FollowJointTrajectory.Goal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES
        q = self.js_joint_position        
        time_from_start = 0

        joint_position_traj = [list(q)]
        joint_velocity_traj = [[0.]*len(q)]

        
        for i, t in enumerate(time):
            if i==0: continue

            prev_frame = misc.pose2KDLframe(poses[i-1])
            frame      = misc.pose2KDLframe(poses[i])
                            
            # ------------------------------------------------------
            # Place your code here
            # ------------------------------------------------------
            # get delta position
            d_p = frame.p - prev_frame.p
            d_position = np.array([d_p[0], d_p[1], d_p[2]]).reshape(3,1)
            
            # get delta orientation
            Rd = np.array([[frame.M[i,j] for j in range(3)] for i in range(3)])
            R  = np.array([[prev_frame.M[i,j] for j in range(3)] for i in range(3)])
            R_delta = Rd @ R.T
            d_w = 0.5 * np.array([[R_delta[2,1] - R_delta[1,2]],
                                [R_delta[0,2] - R_delta[2,0]],
                                [R_delta[1,0] - R_delta[0,1]]])

            # stack into 6-vector twist
            delta = np.vstack([d_position, d_w])
            
            
            # get a jacobian
            J = np.asarray(self.arm_kdl.jacobian(q))

            # get a pseudo inverse of jacobian mtx
            J_inv = J.T @ np.linalg.inv(J @ J.T + 0.1**2 * np.eye(6))

            # get the joint angles
            dq = J_inv @ delta
            q = q + dq[:, 0]

            
            # ------------------------------------------------------
            g.trajectory.points.append(
                JointTrajectoryPoint(positions=q,
                                         velocities=[0]*6,
                                         time_from_start=Duration(sec=int(t),
                                                                      nanosec=int((t-int(t))*1e9) )
                )
                )

            joint_position_traj.append(q.tolist())
            joint_velocity_traj.append(np.array(dq)[:,0].tolist())

            
        self.send_goal(g)
        return time, np.array(joint_position_traj), np.array(joint_velocity_traj), None, None

        
    def move_pose_trajectory(self, poses, duration=5.):
        """ 
        A function that moves the end effector following a pose traj.

        Parameters
        ----------
        pose : list
            a list of Pose objects that include position and orientation pairs
        duration : float
            time duration
        """        
        # Get a sequence of path variables from the min jerk trajectory planning
        time, progress, _, _, _, =\
          mj.min_jerk([0], [1], duration, freq=len(poses)/float(duration))

        # Get a trajectory by interpolating from the start and the goal
        poses_new = []
        for i, pose in enumerate(poses):
            p = Pose()            
            # detach the tool before applying inverse kinematics (if needed)
            p = self.detachTool(p)
            poses_new.append(p)

        g = FollowJointTrajectory.Goal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = JOINT_NAMES
        q = self.js_joint_position        
        time_from_start = 0

        joint_position_traj = [list(q)]
        joint_velocity_traj = [[0.]*len(q)]
        
        for i, t in enumerate(time):
            if i==0: continue

            prev_frame = misc.pose2KDLframe(poses[i-1])
            frame      = misc.pose2KDLframe(poses[i])
                            
            # ------------------------------------------------------
            # Place your code here
            # ------------------------------------------------------
            d_p = frame.p - prev_frame.p
            d_position = np.array([d_p[0], d_p[1], d_p[2]]).reshape(3,1)
            
            # get delta orientation
            Rd = np.array([[frame.M[i,j] for j in range(3)] for i in range(3)])
            R  = np.array([[prev_frame.M[i,j] for j in range(3)] for i in range(3)])
            R_delta = Rd @ R.T
            d_w = 0.5 * np.array([[R_delta[2,1] - R_delta[1,2]],
                                [R_delta[0,2] - R_delta[2,0]],
                                [R_delta[1,0] - R_delta[0,1]]])

            delta = np.vstack([d_position, d_w])
            
            # get a jacobian
            J = np.asarray(self.arm_kdl.jacobian(q))

            # get a pseudo inverse of jacobian mtx
            J_inv = J.T @ np.linalg.inv(J @ J.T + 0.1**2 * np.eye(6))

            # get the joint angles
            dq = J_inv @ delta
            q = q + dq[:, 0]

            
            # ------------------------------------------------------
            g.trajectory.points.append(
                JointTrajectoryPoint(positions=q,
                                         velocities=[0]*6,
                                         time_from_start=Duration(sec=int(t),
                                                                      nanosec=int((t-int(t))*1e9) )
                )
                )

            joint_position_traj.append(q.tolist())
            joint_velocity_traj.append(np.array(dq)[:,0].tolist())
            
        self.send_goal(g)
        return time, np.array(joint_position_traj), np.array(joint_velocity_traj), None, None

        

def problem_1b(arm):
    """Problem 1. B: joint trajectory """
    arm.move_joint([0,0,0,-1.57,0,0])

    while rclpy.ok():
        t, pos, vel, acc, jerk = arm.move_joint([0.4,0,0,-1.57,0,0])
        time.sleep(4.0)
        t, pos, vel, acc, jerk = arm.move_joint([-0.4,0,0,-1.57,0,0])
        time.sleep(4.0)

        # vizualization your trajectory
        arm.get_logger().info("Start to visualize the trajectory. ")
        mj.plot_traj(t, pos, vel, acc, jerk)        

        
def problem_1c(arm):
    """Problem 1. C: position trajectory """

    # Move to the initial configuration
    arm.move_joint([-0.862410612, -1.30713835, 1.31642488, -1.69522468, -1.87213523, 0])
    time.sleep(4.0)
        
    # Move to the goal position
    goal_pose = Pose()
    goal_pose.position.x = 0.4365
    goal_pose.position.y = 0.2
    goal_pose.position.z = 0.2778
    goal_pose.orientation.x = 0.9287
    goal_pose.orientation.y = 0.3342
    goal_pose.orientation.z = -0.1208
    goal_pose.orientation.w = 0.1054
    t, pos, vel, _, _ = arm.move_position(goal_pose, duration=5.)

    mj.plot_traj(t, pos, vel)
    
    
def problem_1d(arm):
    """Problem 1. D: pose trajectory """

    # Move to the initial configuration
    arm.move_joint([0.3754, -1.3828, 1.3637, -1.5517, -1.5706, 0.3754])
    time.sleep(4.0)

    # Move to the goal pose
    goalc_pose = Pose()
    goal_pose.position.x = 0.486899
    goal_pose.position.y = 0.112149-0.3
    goal_pose.position.z = 0.273859
    goal_pose.orientation.x = 0.7071 
    goal_pose.orientation.y = 0.7071
    goal_pose.orientation.z = 0.0
    goal_pose.orientation.w = 0.0
    
    t, pos, vel, _, _ = arm.move_pose(goal_pose, duration=5.)

    mj.plot_traj(t, pos, vel)


    
        
def main(args=None):
    """ a main function """
    args = get_args(sysargv=sys.argv)[0]
    
    rclpy.init() 
    
    # create action client
    arm = ArmClient()

    # Comment in/out the function you want
    if args.prob_1b: problem_1b(arm)
    elif args.prob_1c: problem_1c(arm)
    elif args.prob_1d: problem_1d(arm)
    
    #arm.destroy_node()
    del arm
    rclpy.shutdown()
    

def get_args(sysargv=None):
    p = argparse.ArgumentParser(
        description="Run the move_joint script for assignment 3")
    p.add_argument('--1b', action='store_true', dest='prob_1b',
                 help='use this option if you want to run the script for the problem 1b')
    p.add_argument('--1c', action='store_true', dest='prob_1c',
                 help='use this option if you want to run the script for the problem 1c')
    p.add_argument('--1d', action='store_true', dest='prob_1d',
                 help='use this option if you want to run the script for the problem 1d')
    argv = sysargv[1:] if sysargv is not None else sys.argv[1:]
    args = p.parse_known_args(argv)
    return args

    
if __name__ == '__main__':
    main()
        

    