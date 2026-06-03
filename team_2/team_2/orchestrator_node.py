#!/usr/bin/env python3
"""
orchestrator_node.py
"""
import copy
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from tf2_ros import (Buffer, TransformListener, LookupException,
                     ConnectivityException, ExtrapolationException)
import tf2_geometry_msgs

from team_2_interfaces.srv import DetectObjects
from riro_srvs.srv import StringPose          # reused for the bbox -> 3D lift

# --- SEAM 4: match these to your working grasp code --------------------------
from assignment_2 import move_joint as mj      # provides ArmClient
from manip_challenge import move_gripper
# -----------------------------------------------------------------------------
import math
from geometry_msgs.msg import Pose, Quaternion
from manip_challenge import get_joint 

def quat_mul(a, b):
    """Hamilton product a ⊗ b (ROS x,y,z,w convention)."""
    q = Quaternion()
    q.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    q.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
    q.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
    q.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    return q

def quat_about_z(angle):
    q = Quaternion()
    q.z = math.sin(angle/2.0)
    q.w = math.cos(angle/2.0)
    return q

def make_pose(x, y, z, roll, pitch, yaw):
    """(x,y,z) metres, (roll,pitch,yaw) radians, all in base_link."""
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.x = sr * cp * cy - cr * sp * sy
    p.orientation.y = cr * sp * cy + sr * cp * sy
    p.orientation.z = cr * cp * sy - sr * sp * cy
    p.orientation.w = cr * cp * cy + sr * sp * sy
    return p

# --- POSITIONS ----------------------------------------------------------------
# x, y, z, roll, pitch, yaw (radians)
HOME_POSES = [
    (0.55,  0.35, 0.60, -math.pi, 0.0, 0.0),
    (0.55,  0.05, 0.60, -math.pi, 0.0, 0.0),
    (0.55, -0.20, 0.60, -math.pi, 0.0, 0.0),
]

# SCOUT_POSE   = (-0.20, -0.05, 0.45, 3 * math.pi/4, 0.0, math.pi/2)

# PRE_SHELF_POSES = [
#     # (1.15, 0.3, 0.05, math.pi/2, 0.0, 0.0),
#     # (1.15, 0.3, 0.30, math.pi/2, 0.0, 0.0),
#     (0.60, -0.33, 0.50, math.pi/2, 0.0, math.pi/2)
# ]


# POST_SHELF_POSES = [
#     (1.15, -0.35, 0.05, math.pi/2, 0.0, 0.0),
#     (1.15, -0.35, 0.30, math.pi/2, 0.0, 0.0),
#     (1.00, -0.33, 0.50, math.pi/2, 0.0, math.pi/2)
# ]

# STORAGE_POSES = {
#     'left_storage':  (-0.05,  0.60, 0.55, math.pi, 0.0, 0.0),
#     'right_storage': (-0.05, -0.60, 0.55, math.pi, 0.0, 0.0)
# }

HOME_POSES_JOINTS = [
                [0.3562841841001843, -0.9628604728892204, 0.011549683289610273, -0.632696892891174, -1.5702528606004802, 1.9182459455950602],
                [-0.09760216271339925, -1.041465208208678, 0.007976593885553171, -0.5451334190770651, -1.570210237498598, 1.4776046285377251],
                [-0.5355024046855043, -1.0059814986612452, 0.002029311252828683, -0.5756700956254002, -1.5698330212301568, 1.0396796225880653]
            ]         
SCOUT_POSE_JOINTS = [0.017202321595274267, -2.8693023874037227, 1.4228273057359349, -0.9013666354949029, -1.5966035125312512, 0.012271264979222866]

STORAGE_POSES_JOINTS = {
                    'left_storage': [1.3353573876166518, -1.348273758932529, 0.6411880748192984, -0.874215158899556, -1.5737074037209007, 2.8773348290953638],
                    'right_storage': [-1.7935734073518597, -1.2752218287662815, 0.5669301759815633, -0.8697850266449945, -1.576582165253679, -0.22061197194364868]
                    }

PRE_SHELF_POSES_JOINTS = [
                    [0.17285882419207604, 0.009414032585641493, 0.04142618024092783, -1.2309130438531848, -3.1167689924260484, 1.9431021777750577],
                    [0.1868212461565257, -0.21137538666348565, -0.022417456928694304, -1.3048437797960213, -3.113571353761288, 1.609871180198019],
                    [-0.9110161564253627, -1.4206915427165185, 1.4838585428541253, -3.135808794737544, -0.6651189437124587, -0.0566975763987704]
                ]

POST_SHELF_POSES_JOINTS = [
    [-0.1001950339110103, 0.03066953328747561, -0.00021623515338185738, -1.5241709836884587, -3.1503408239070074, 1.6483366493474607],
    [-0.11255248968737187, -0.17735367529810056, -0.006038652614095945, -1.577814610424966, -3.12448853397301, 1.3799779404378971],
    [-0.5558515282526402, -0.45263417722776306, 0.17613617529742698, -2.8599554581953335, -1.0168549357011385, -0.0028841526061399313]
]

# Oriented-grasp convention: once we know the object's long-axis yaw in
# base_link, the jaws must close ACROSS it (perpendicular) -> +90°. If grasps
# come out rotated 90° (jaws along the long axis instead), flip this to 0.0.
# One-time calibration; depends on which way the borrowed FK gripper opens.
GRIPPER_YAW_OFFSET = 0.0
# -----------------------------------------------------------------------------


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter('camera_optical_frame', 'wrist_camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('settle_sec', 0.8)   # let the camera stream catch up to the new arm pose
        self.declare_parameter('time_budget_sec', 300.0)

        self.optical_frame = self.get_parameter('camera_optical_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.settle = float(self.get_parameter('settle_sec').value)
        self.time_budget = float(self.get_parameter('time_budget_sec').value)

        self.tasks = None  # set when /parsed_tasks arrives -> leaves standby

        # Arm owner.
        self.arm = mj.ArmClient()

        # TF (camera optical -> base_link), populated while we spin.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service clients.
        self.detect_cli = self.create_client(DetectObjects, self.get_parameter('detect_service').value)
        self.lift_cli = self.create_client(StringPose, self.get_parameter('lift_service').value)

        # Latched subscription so a parser that published before we were up
        # still delivers (TRANSIENT_LOCAL must match the parser's publisher QoS).
        tasks_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, self.get_parameter('parsed_tasks_topic').value,
            self.on_tasks, tasks_qos)

        self.get_logger().info('orchestrator_node: created.')

        self.truth_cli = self.create_client(StringPose, '/get_object_pose')
        self._wm_cache = []
        self.create_subscription(String, 'world_model', self._dbg_wm, 10)

        self.shelf_layer_occupied = [0, 0, 0] # there are three layers on the shelf, track how many objects are in each to know where to place the next one

    def _dbg_wm(self, msg):
        try:
            self._wm_cache = json.loads(msg.data).get('world', [])
        except Exception as e:
            self.get_logger().warn(f"world_model parse failed: {e}")

    def get_object_truth(self, label):
        """DEBUG ONLY. Ground truth straight from the /world_model topic"""
        label = label.strip().lower().replace(' ', '_')
        for obj in self._wm_cache:
            if obj.get('name', '').lower().startswith(label):
                x, y, z = obj['pose'][0], obj['pose'][1], obj['pose'][2]
                return obj['pose']
        self.get_logger().warn(f"No world-model entry starting with '{label}'.")
        return None

    # ---- standby trigger ----------------------------------------------------
    def on_tasks(self, msg):
        try:
            self.tasks = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Bad /parsed_tasks JSON: {e}')

    # ---- service helpers ----------------------------------------------------
    def wait_for_services(self):
        for cli, name in ((self.detect_cli, 'detect_objects'),
                          (self.lift_cli, 'lift_bbox_to_3d')):
            while rclpy.ok() and not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for '{name}' service...")

    def call_perception(self, names):
        req = DetectObjects.Request()
        req.data = json.dumps(names)
        fut = self.detect_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            return []
        try:
            return json.loads(res.data).get('detections', [])
        except json.JSONDecodeError:
            return []

    def call_lift(self, bbox):
        """bbox = [x1,y1,x2,y2] pixels -> 3D Pose in the camera optical frame,
        or None if the grasp server found no valid depth in the box."""
        req = StringPose.Request()
        req.data = json.dumps(bbox)
        fut = self.lift_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            return None
        p = res.pose
        # The grasp server returns an empty Pose (all zeros) on failure.
        if p.position.x == 0.0 and p.position.y == 0.0 and p.position.z == 0.0:
            return None
        return p

    def detect_and_lift(self, names, candidate_tasks):
        """One detect -> pick -> lift -> TF-to-base_link cycle from the CURRENT
        arm pose.

        Returns (det, task, pose_opt, obj_bl) or None on any failure.
          pose_opt : Pose in the camera optical frame (what grasp_and_place wants;
                     its .orientation carries the OBB long-axis from grasp_server)
          obj_bl   : same point in base_link (for the home-bucket decision/debug)
        """
        dets = self.call_perception(names)
        if not dets:
            self.get_logger().warn('No detections.')
            return None

        pair = self.pick_target(dets, candidate_tasks)
        if pair is None:
            self.get_logger().warn('No detection matched a candidate task.')
            return None
        det, task = pair

        pose_opt = self.call_lift(det['bbox'])
        if pose_opt is None:
            self.get_logger().warn('Lift-to-3D failed (no valid depth in bbox).')
            return None

        obj_bl = self.transform_pose(pose_opt, self.optical_frame, self.base_frame)
        if obj_bl is None:
            return None

        return det, task, pose_opt, obj_bl

    # ---- TF -----------------------------------------------------------------
    def transform_pose(self, pose, source_frame, target_frame):
        try:
            ps = PoseStamped()
            ps.header.frame_id = source_frame
            ps.pose = pose
            out = self.tf_buffer.transform(
                ps, target_frame, timeout=Duration(seconds=1.0))
            return out.pose
        except (LookupException, ConnectivityException,
                ExtrapolationException) as e:
            self.get_logger().error(f'TF transform failed: {e}')
            return None

    def wait_for_tf(self):
        """Spin the node so the TF listener fills the buffer (we are not
        continuously spun, so without this can_transform never becomes true)."""
        while rclpy.ok():
            if self.tf_buffer.can_transform(
                    self.base_frame, self.optical_frame,
                    rclpy.time.Time(), timeout=Duration(seconds=1.0)):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    # ---- target selection ---------------------------------------------------
    @staticmethod
    def pick_target(detections, remaining):
        """Highest-confidence detection whose object matches a remaining task."""
        best = None
        for det in detections:
            d_obj = det['object'].strip().lower()
            for task in remaining:
                if task['object'].strip().lower() == d_obj:
                    if best is None or det['conf'] > best[0]['conf']:
                        best = (det, task)
        return best

    @staticmethod
    def select_home_pose(obj_y):
        """Index of the HOME_POSE whose y is nearest the object's y, so the wrist
        camera ends up looking roughly straight down at the target."""
        ys = [p[1] for p in HOME_POSES]
        return min(range(len(HOME_POSES)), key=lambda i: abs(ys[i] - obj_y))

    # ---- OBB orientation -> gripper yaw -------------------------------------
    def _oriented_grasp_yaw(self, pose_opt):
        """Decode the object's long-axis angle (grasp_server packs it into
        pose_opt.orientation as a rotation about optical +Z), snap to one of four
        image-space buckets, and return (bucket_name, wrist3_delta_rad).

        The delta goes STRAIGHT into rotate_gripper(): the wrist camera rotates WITH
        wrist_3, so an angle measured in the image is already a wrist_3 increment --
        no base_link TF needed. GRIPPER_YAW_OFFSET turns the jaws to close ACROSS
        (perpendicular to) the long axis.
        """
        q = pose_opt.orientation
        angle = 2.0 * math.atan2(q.z, q.w)        # long-axis angle in optical XY
        a = angle % math.pi
        cats = [(0.0,               'horizontal'),
                (math.pi / 4.0,     'left-to-right'),
                (math.pi / 2.0,     'vertical'),
                (3 * math.pi / 4.0, 'right-to-left')]

        def circ(x, b):
            d = abs(x - b) % math.pi
            return min(d, math.pi - d)

        a_snap, name = min(cats, key=lambda c: circ(a, c[0]))

        # Across the long axis (+offset), folded into (-pi/2, pi/2] so the wrist
        # takes the short way round (jaws are symmetric under 180 deg -> no joint-limit swings).
        delta = a_snap + GRIPPER_YAW_OFFSET
        delta = (delta + math.pi / 2.0) % math.pi - math.pi / 2.0
        return name, delta

    # ---- arm moves ----------------------------------------------------------        
    def move_to_scout(self):
        self.get_logger().info('Moving to SCOUT pose...')
        self.arm.move_joint(SCOUT_POSE_JOINTS)

    def rotate_gripper(self, angle):
        self.get_logger().info(f'Rotating gripper by {math.degrees(angle):.0f} deg...')
        cur = list(get_joint.get_joint_angles(self))
        cur[5] += angle
        self.arm.move_joint(cur)

    def move_pose_by_joints(self, target):
        q = self.arm.pose_to_joints(target)
        if q is None:
            self.get_logger().warn('approach: IK None -> move_pose fallback')
            self.arm.move_pose(target)
        else:
            self.arm.move_joint(q)

    def grasp_and_place(self, obj_pose_optical, destination):
        """Adapted from example5 ObjectGraspClient.request_pick, plus a
        destination-aware place."""
        move_gripper.gripper_open(self)

        if not self.wait_for_tf():
            self.get_logger().error('TF never became available.')
            return False

        obj_bl = self.transform_pose(obj_pose_optical, self.optical_frame, self.base_frame)

        # ---1. approach---
        goal = copy.copy(obj_bl)
        goal.position.z += 0.10
        self.move_pose_by_joints(goal)
            
        # ---2. spin the jaws of the gripper in place---
        cat_name, grasp_yaw = self._oriented_grasp_yaw(obj_pose_optical)
        self.get_logger().info(f'OBB: {cat_name} -> wrist_3 += {math.degrees(grasp_yaw):+.0f} deg')
        self.rotate_gripper(grasp_yaw)

        # ---3. descend, keeping the rotated orientation---
        cur = list(get_joint.get_joint_angles(self))
        goal.orientation = self.arm.fk_request(cur).orientation
        goal.position.z -= 0.13
        self.arm.move_pose(goal)

        # ---4. grasp---
        move_gripper.gripper_close(self, force=0.5, gripper_close_pos=0.5)

        # ---5. lift clear---
        goal.position.z += 0.40
        self.arm.move_pose(goal)

        # ---6. carry to the storage and release---
        # if shelf, moving to the pre-place pose is necessary
        if destination == 'shelf':
            # calculate which layer of the shelf to place on based on how many objects are already there
            min_count = min(self.shelf_layer_occupied)
            layer_idx = next(
                idx for idx in reversed(range(len(self.shelf_layer_occupied)))
                if self.shelf_layer_occupied[idx] == min_count
            )
            self.arm.move_joint(PRE_SHELF_POSES_JOINTS[layer_idx])
            self.arm.move_joint(POST_SHELF_POSES_JOINTS[layer_idx])
        else:
            self.arm.move_joint(STORAGE_POSES_JOINTS[destination])

        move_gripper.gripper_open(self)
        
        # if shelf, move back to the pre-place pose to avoid collisions and update the shelf layer occupation count
        if destination == 'shelf':
            self.shelf_layer_occupied[layer_idx] += 1
            self.arm.move_joint(PRE_SHELF_POSES_JOINTS[layer_idx])

        return True

    # ---- one-time survey ----------------------------------------------------
    def coarse_survey(self, tasks):
        """From the CURRENT (scout) pose, detect every requested object type once,
        lift each detection to a rough base_link position, and assign each task to
        the nearest-in-y HOME_POSES bucket.

        Returns (plan, missed):
          plan   : list of (task, home_idx), one per task we could place a bucket on
          missed : list of object names a task wanted but the scout never located

        We only keep the base_link y here (time-invariant, safe to remember).
        The actual grasp pose is re-measured later at the home pose, NOT reused
        from this survey -- an optical-frame pose from scout is meaningless once
        the arm has moved.
        """
        names = sorted({t['object'].strip().lower() for t in tasks})
        dets = self.call_perception(names)

        # Lift each detection once -> a located physical object with a coarse y.
        located = []   # [{'obj': str, 'y': float, 'claimed': bool}, ...]
        for det in dets:
            pose_opt = self.call_lift(det['bbox'])
            if pose_opt is None:
                continue
            bl = self.transform_pose(pose_opt, self.optical_frame, self.base_frame)
            if bl is None:
                continue
            idx = self.select_home_pose(bl.position.y)
            located.append({'obj': det['object'].strip().lower(),
                            'y': bl.position.y, 'claimed': False})
            self.get_logger().info(f"======={det['object']} belongs to the HOME_POSES[{idx}]======")

        # Pair each task with an unclaimed located object of the same type, so
        # duplicates of one type get spread across their real buckets.
        plan, missed = [], []
        for task in tasks:
            obj = task['object'].strip().lower()
            cand = next((L for L in located if not L['claimed'] and L['obj'] == obj), None)
            if cand is None:
                missed.append(obj)
                continue
            cand['claimed'] = True
            plan.append((task, self.select_home_pose(cand['y'])))
        return plan, missed

    # ---- main loop ----------------------------------------------------------
    def run_contest(self):
        raw = self.tasks
        tasks = raw['tasks'] if isinstance(raw, dict) and 'tasks' in raw else raw

        t0 = time.time()

        # --- SURVEY ONCE from the scout pose ---------------------------------
        plan, missed = self.coarse_survey(tasks)

        if missed:
            self.get_logger().warn(
                f'Not located at scout (will be skipped): {sorted(set(missed))}')
        if not plan:
            self.get_logger().warn('Nothing localized from scout; nothing to do.')
            return

        # Bucket order keeps same-home objects adjacent in the logs/run.
        plan.sort(key=lambda tp: tp[1])
        self.get_logger().info(
            'Plan: ' + ', '.join(f"{t['object']}->HOME[{i}]" for t, i in plan))

        # --- Pick each planned object from its assigned home pose ------------
        for task, home_idx in plan:
            if (time.time() - t0) >= self.time_budget:
                self.get_logger().warn('Time budget exhausted; stopping.')
                break

            self.get_logger().info(f"=== {task['object']} -> {task['destination']} from HOME_POSES[{home_idx}] ===")
            self.get_logger().info(f'Moving to HOME_POSES[{home_idx}]...')
            self.arm.move_joint(HOME_POSES_JOINTS[home_idx])
            time.sleep(self.settle)

            # Fresh near-top-down localize for THIS object (accurate grasp pose).
            acc = self.detect_and_lift([task['object'].strip().lower()], [task])
            if acc is None:
                self.get_logger().warn(
                    f"Couldn't acquire {task['object']} from HOME_POSES[{home_idx}]; "
                    'skipping (no re-scout in this design).')
                continue
            _, _, pose_opt, obj_bl = acc

            # One attempt; not retried on failure (survey-once design).
            self.grasp_and_place(pose_opt, task['destination'].strip().lower())

        elapsed = time.time() - t0
        self.get_logger().info(f'Contest loop ended after {elapsed:.1f}s.')


def main():
    rclpy.init()
    node = Orchestrator()
    node.wait_for_services()

    node.move_to_scout()   

    node.get_logger().info('STANDBY: waiting for /parsed_tasks ...')
    while rclpy.ok() and node.tasks is None:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.tasks is not None:
        node.run_contest()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()