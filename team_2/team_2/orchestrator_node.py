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
    (0.55,  0.4, 0.60, -math.pi, 0.0, 0.0),
    (0.55,  0.05, 0.60, -math.pi, 0.0, 0.0),
    (0.55, -0.20, 0.60, -math.pi, 0.0, 0.0),
]

# SCOUT_POSE   = (-0.20, -0.05, 0.45, 3 * math.pi/4, 0.0, math.pi/2)

# PRE_SHELF_POSES = [
#     (1.15, 0.3, 0.05, math.pi/2, 0.0, 0.0),
#     (1.15, 0.3, 0.30, math.pi/2, 0.0, 0.0),
#     (0.60, -0.33, 0.50, math.pi/2, 0.0, math.pi/2)
# ]


# POST_SHELF_POSES = [
#     (1.15, -0.32, 0.025, math.pi/2+0.3, 0.0, 0.0),
#     (1.15, -0.32, 0.28, math.pi/2+0.3, 0.0, 0.0),
#     (1.00, -0.33, 0.52, math.pi/2+0.4, 0.0, math.pi/2)
# ]


# STORAGE_POSES = {
#     'left_storage':  (-0.05,  0.60, 0.25, math.pi, 0.0, 0.0),
#     'right_storage': (-0.05, -0.60, 0.25, math.pi, 0.0, 0.0)
# }

HOME_POSES_JOINTS = [
                [0.4486239282397744, -0.919427693246301, -0.0009643325054386759, -0.6664451503075698, -1.5698670094956702, 2.0186363613790976],
                [-0.09760216271339925, -1.041465208208678, 0.007976593885553171, -0.5451334190770651, -1.570210237498598, 1.4776046285377251],
                [-0.5355024046855043, -1.0059814986612452, 0.002029311252828683, -0.5756700956254002, -1.5698330212301568, 1.0396796225880653]
            ]         
SCOUT_POSE_JOINTS = [0.017202321595274267, -2.8693023874037227, 1.4228273057359349, -0.9013666354949029, -1.5966035125312512, 0.012271264979222866]

STORAGE_POSES_JOINTS = {
                    'left_storage': [1.3262682423335923, -1.5261878782927405, 1.5006351440038201, -1.5529224614769819, -1.5643691270548534, 2.868371580877126],
                    'right_storage': [-1.7741962334802774, -1.4659544651890226, 1.4349766582912324, -1.5319499488171424, -1.576430258078094, -0.20136340291521987]
                    }
# 

PRE_SHELF_POSES_JOINTS = [
                    [0.17285882419207604, 0.009414032585641493, 0.04142618024092783, -1.2309130438531848, -3.1167689924260484, 1.9431021777750577],
                    [0.1868212461565257, -0.21137538666348565, -0.022417456928694304, -1.3048437797960213, -3.113571353761288, 1.609871180198019],
                    [-0.9110161564253627, -1.4206915427165185, 1.4838585428541253, -3.135808794737544, -0.6651189437124587, -0.0566975763987704]
                ]

POST_SHELF_POSES_JOINTS = [
    [-0.13535970323704163, 0.023049948383443927, -0.04203316177540594, -1.750520392646267, -2.850942302274265, 1.3479522636327192],
    [-0.1582175292289698, -0.20618354710604558, -0.0364649370185775, -1.6509244782416772, -2.8357651870951144, 1.2233524083046359],
    [-0.5504861307885196, -0.6772529340569791, 0.37828048434279177, -2.39458452572829, -1.0650724776737845, -0.22683857683566672]
]

GRIPPER_YAW_OFFSET = math.pi / 2.0
# -----------------------------------------------------------------------------


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter('camera_optical_frame', 'wrist_camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('max_pick_passes', 3)
        self.declare_parameter('elbow_floor_z', 0.20)

        self.optical_frame = self.get_parameter('camera_optical_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.max_pick_passes = int(self.get_parameter('max_pick_passes').value) 
        self.elbow_floor_z = float(self.get_parameter('elbow_floor_z').value)  # <-- add

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
        self.create_subscription(String, self.get_parameter('parsed_tasks_topic').value, self.on_tasks, tasks_qos)

        self.get_logger().info('orchestrator_node: created.')

        self.shelf_layer_occupied = [0, 0, 0] # there are three layers on the shelf, track how many objects are in each to know where to place the next one


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

    def call_perception(self, names, camera='wrist'):
        """Request detection from specified camera ('wrist' or 'scene')."""
        req = DetectObjects.Request()
        req.data = json.dumps({'classes': names, 'camera': camera})
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

        return pose_opt

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

        self.get_logger().info(f'OBB: {name} -> wrist_3 += {math.degrees(delta):+.0f} deg')

        return delta

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
        """Move the EE to `target` in joint space on the elbow-up branch."""
        q = self._solve_ik_elbow_up(target)
        if q is None:
            return False
        self.arm.move_joint(q)
        return True

    def _forearm_z(self, q):
        """Elbow height (forearm_link origin z) in base_link at config q, via
        the arm's own KDL chain -- this is what distinguishes the UR5's
        elbow-up branch from the elbow-down one that drives the forearm into
        the table."""
        try:
            T = np.asarray(self.arm.arm_kdl.forward(q, end_link='forearm_link'))
            return float(T[2, 3])
        except Exception as e:
            self.get_logger().warn(f'_forearm_z: FK to forearm_link failed ({e}).')
            return None   # can't gate -> caller trusts the raw solve

    def _solve_ik_elbow_up(self, target):
        """Joints that reach `target` on the elbow-UP branch. pose_to_joints
        lands on whatever branch its seed + Cartesian path drift into, so we
        solve, FK-check the elbow, and on a fold retry from seeds biased toward
        elbow-up (and finally the curated HOME configs). Returns elbow-up
        joints, or None if no seed produced one."""
        cur = list(get_joint.get_joint_angles(self))

        seeds = [cur]
        for d in (0.4, 0.8):                 # shoulder_lift up + elbow folded in
            s = list(cur); s[1] -= d; s[2] += d
            seeds.append(s)
        seeds.extend(HOME_POSES_JOINTS)      # known elbow-up fallbacks

        best = None
        for seed in seeds:
            q = self.arm.pose_to_joints(target, q_seed=seed)
            if q is None:
                continue
            z = self._forearm_z(q)
            if z is None:                    # FK gate unavailable -> trust solve
                return q
            if best is None:
                best = (q, z)
            if z >= self.elbow_floor_z:
                self.get_logger().info(f'IK elbow-up: forearm z={z:.3f}')
                return q
        if best is not None:
            self.get_logger().warn(
                f'IK: no elbow-up branch (best forearm z={best[1]:.3f} '
                f'< floor={self.elbow_floor_z:.3f}).')
        return None

    def grasp_and_place(self, obj_pose_optical, destination):
        """Adapted from example5 ObjectGraspClient.request_pick, plus a
        destination-aware place."""
        move_gripper.gripper_open(self)

        if not self.wait_for_tf():
            self.get_logger().error('TF never became available.')
            return False

        neutral_opt = Pose()
        neutral_opt.position = obj_pose_optical.position
        neutral_opt.orientation.w = 1.0          # identity: no long-axis spin
        obj_bl = self.transform_pose(neutral_opt, self.optical_frame, self.base_frame)

        # ---1. approach---
        goal = copy.copy(obj_bl)
        goal.position.z += 0.10
        self.move_pose_by_joints(goal)
            
        # ---2. spin the jaws of the gripper in place---
        self.rotate_gripper(self._oriented_grasp_yaw(obj_pose_optical))

        # ---3. descend, keeping the rotated orientation---
        cur = list(get_joint.get_joint_angles(self))
        goal.orientation = self.arm.fk_request(cur).orientation
        goal.position.z -= 0.12
        self.move_pose_by_joints(goal)

        # ---4. grasp---
        move_gripper.gripper_close(self, force=0.5, gripper_close_pos=0.55)

        # ---5. lift up to the HOME POSE 1---
        self.arm.move_joint(HOME_POSES_JOINTS[1]) # move to a middle home pose before going to the shelf to avoid collisions with the shelf when moving from the lift pose

        # ---6. carry to the storage and release---
        if destination == 'shelf':
            # calculate which layer of the shelf to place on based on how many objects are already there
            min_count = min(self.shelf_layer_occupied)
            layer_idx = next(
                idx for idx in reversed(range(len(self.shelf_layer_occupied)))
                if self.shelf_layer_occupied[idx] == min_count
            )
            self.arm.move_joint(PRE_SHELF_POSES_JOINTS[layer_idx]) # move to the pre-place pose for the correct layer
            self.arm.move_joint(POST_SHELF_POSES_JOINTS[layer_idx]) # move to the place pose for the correct layer
        else:
            self.arm.move_joint(STORAGE_POSES_JOINTS[destination])

        move_gripper.gripper_open(self)
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

        If objects are missed on primary wrist-camera scout, try secondary detection
        using the scene (overhead) camera from the same arm position for better accuracy.

        Returns (plan, missed):
          plan   : list of (task, home_idx), one per task we could place a bucket on
          missed : list of object names a task wanted but the scout never located

        We only keep the base_link y here (time-invariant, safe to remember).
        The actual grasp pose is re-measured later at the home pose, NOT reused
        from this survey -- an optical-frame pose from scout is meaningless once
        the arm has moved.
        """
        names = sorted({t['object'].strip().lower() for t in tasks})
        
        # Primary detection: wrist camera from scout pose (oblique angle)
        dets = self.call_perception(names, camera='wrist')

        # Lift each detection once -> a located physical object with a coarse y.
        located = []   # [{'obj': str, 'y': float, 'claimed': bool, 'camera': str}, ...]
        for det in dets:
            pose_opt = self.call_lift(det['bbox'])
            if pose_opt is None:
                continue
            bl = self.transform_pose(pose_opt, self.optical_frame, self.base_frame)
            if bl is None:
                continue
            idx = self.select_home_pose(bl.position.y)
            located.append({'obj': det['object'].strip().lower(),
                            'y': bl.position.y, 'claimed': False, 'camera': 'wrist'})

        # Pair each task with an unclaimed located object of the same type.
        plan, missed = [], []
        for task in tasks:
            obj = task['object'].strip().lower()
            cand = next((L for L in located if not L['claimed'] and L['obj'] == obj), None)
            if cand is None:
                missed.append(obj)
                continue
            cand['claimed'] = True
            plan.append((task, self.select_home_pose(cand['y'])))
        
        # If we missed objects, try secondary detection using scene camera (overhead view, same arm position)
        if missed:
            self.get_logger().info(f'Attempting secondary scout (scene camera) for missed: {missed}')
            dets_scene = self.call_perception(missed, camera='scene')
            
            for det in dets_scene:
                obj_lower = det['object'].strip().lower()
                # Skip if already located at primary scout (wrist camera).
                if any(L['obj'] == obj_lower and not L['claimed'] for L in located):
                    continue
                    
                pose_opt = self.call_lift(det['bbox'])
                if pose_opt is None:
                    continue
                bl = self.transform_pose(pose_opt, self.optical_frame, self.base_frame)
                if bl is None:
                    continue
                idx = self.select_home_pose(bl.position.y)
                located.append({'obj': obj_lower, 'y': bl.position.y, 'claimed': False, 'camera': 'scene'})
                self.get_logger().info(
                    f'[SCENE CAMERA] {det["object"]} -> HOME[{idx}] (confidence: {det.get("conf", "N/A")})')
                # Retry plan for this object.
                for task in tasks:
                    if task['object'].strip().lower() == obj_lower:
                        missed.remove(obj_lower) if obj_lower in missed else None
                        plan.append((task, idx))
                        break
        
        return plan, missed

    # ---- main loop ----------------------------------------------------------
    def run_contest(self):
        raw = self.tasks
        tasks = raw['tasks'] if isinstance(raw, dict) and 'tasks' in raw else raw

        t0 = time.time()

        for pass_idx in range(self.max_pick_passes):
            # ----- 1. From the scout pose, detect objects -----
            plan, missed = self.coarse_survey(tasks)

            if missed:
                self.get_logger().info(f'[Pass {pass_idx + 1}] not present at scout: {sorted(set(missed))}')

            if not plan:
                if pass_idx == 0:
                    self.get_logger().warn('Nothing localized from scout; nothing to do.')
                else:
                    self.get_logger().info('All done! Re-scout found no remaining instructed objects.')
                break

            plan.sort(key=lambda tp: tp[1])
            self.get_logger().info(f'[Pass {pass_idx + 1}] Plan: ' + ', '.join(f"{t['object']}->HOME[{i}]" for t, i in plan))

            # ----- 2. Pick each planned object from its assigned home pose -----
            for task, home_idx in plan:
                self.get_logger().info(f"===== {task['object']} -> {task['destination']} from HOME_POSES[{home_idx}] =====")
                self.arm.move_joint(HOME_POSES_JOINTS[home_idx])

                # Fresh near-top-down localize for THIS object (accurate grasp pose).
                pose_opt = self.detect_and_lift([task['object'].strip().lower()], [task])
                if pose_opt is None:
                    self.get_logger().warn(
                        f"Couldn't acquire {task['object']} from HOME_POSES[{home_idx}]; "
                        'skipping (will be re-checked next pass).')
                    continue

                self.grasp_and_place(pose_opt, task['destination'].strip().lower())

            self.move_to_scout()

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