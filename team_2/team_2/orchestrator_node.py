#!/usr/bin/env python3
"""
orchestrator_node.py

The ONLY node that commands the arm. Everything else (perception, 3D lift) is a
service it calls. This is what makes the architecture safe: one owner of the
controller, so two nodes never fight over trajectories.

Flow (this is the competition flow):

    STANDBY  ── wait for one /task_commands-derived message on /parsed_tasks
       │         (latched, so a parser that published first still reaches us)
       ▼
    LOOP while tasks remain and time budget left:
       move arm to SCOUT_POSE  (oblique view — the whole point: classify from
                                a viewpoint that shows labels/shape, not top-down)
       ── call 'detect_objects'      (perception service, 2D boxes)
       ── pick the best detection that matches a remaining task
       ── call 'lift_bbox_to_3d'     (grasp-server service, bbox -> 3D Pose in
                                       wrist_camera_color_optical_frame)
       ── TF that pose into base_link, then grasp + place at the task's storage
       ── drop the completed task

Concurrency model: SINGLE-THREADED, on-demand spin — exactly like
example5_grasp_client. We are NOT continuously spun by an executor, so blocking
service/action calls (spin_until_future_complete, ArmClient's internal spins)
don't deadlock. The only place we must spin manually is while waiting for TF.

============================  SEAMS YOU MUST FILL  ============================
  1. SCOUT_POSE      — six joint angles for the oblique scouting view. Find them
                       by jogging the arm and reading get_joint.get_joint_angles.
  2. HOME_POSE       — optional safe start pose.
  3. PLACE_POSES     — a joint config per destination string where the arm opens
                       the gripper to drop the object.
  4. The import of ArmClient / move_gripper — point these at whatever your
     WORKING grasp code uses. example5 uses `assignment_2.move_joint.ArmClient`
     and `manip_challenge.move_gripper`; example1 uses `manip_challenge.move_*`.
  5. 'lift_bbox_to_3d' service — see note at bottom; it's a ~6-line change to
     example5_grasp_server (strip Gemini, read bbox from request.data).
==============================================================================
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
import tf2_geometry_msgs  # noqa: F401  (registers Pose transforms with tf2)

from team_2_interfaces.srv import DetectObjects
from riro_srvs.srv import StringPose          # reused for the bbox -> 3D lift

# --- SEAM 4: match these to your working grasp code --------------------------
from assignment_2 import move_joint as mj      # provides ArmClient
from manip_challenge import move_gripper
# -----------------------------------------------------------------------------
import math
from geometry_msgs.msg import Pose

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

# --- POSITIONS ---------------------------------------
# x, y, z, roll, pitch, yaw (in radians)
HOME_POSE  = (0.45, 0.15, 0.55, math.pi, 0.0, 0.0)
SCOUT_POSE   = (0.95, -0.3, 0.50, math.pi/2, 0.0, math.pi/2)

PRE_SHELF_POSES = [
    (1.15, 0.2, 0.05, math.pi/2, 0.0, 0.0),
    (1.15, 0.2, 0.30, math.pi/2, 0.0, 0.0),
    (0.95, 0.2, 0.50, math.pi/2, 0.0, math.pi/2)
]

STORAGE_POSES = {
    'left_storage':  (0.00,  0.55, 0.60, math.pi, 0.0, 0.0),
    'right_storage': (0.00, -0.55, 0.60, math.pi, 0.0, 0.0),
}
# -----------------------------------------------------------------------------


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter('camera_optical_frame', 'wrist_camera_color_optical_frame') #TODO can change to wrist_camera_color_optical_frame if using wrist camera for perception
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

    def grasp_and_place(self, obj_pose_optical, destination):
        """Adapted from example5 ObjectGraspClient.request_pick, plus a
        destination-aware place."""
        move_gripper.gripper_open(self)

        # Borrow a downward gripper orientation from FK of a known-good config.
        pose_bl_to_gripper = self.arm.fk_request([0.0, -np.pi / 2.0, 1.0, -np.pi / 3.0, -np.pi / 2.0, 0.0])

        if not self.wait_for_tf():
            self.get_logger().error('TF never became available.')
            return False
        obj_bl = self.transform_pose(obj_pose_optical, self.optical_frame, self.base_frame)
        if obj_bl is None:
            return False

        goal = copy.copy(obj_bl)
        goal.orientation = pose_bl_to_gripper.orientation

        # 1. approach from above
        # goal.position.x -= 0.02
        goal.position.z += 0.10
        self.arm.move_pose(goal)

        # 2. descend onto the object
        goal.position.z -= 0.13
        self.arm.move_pose(goal)
        # 3. grasp
        move_gripper.gripper_close(self, force=0.5, gripper_close_pos=0.5)
        # 4. lift clear
        goal.position.z += 0.20
        self.arm.move_pose(goal)

        # 5. carry to the storage and release
        if destination == 'shelf':
            # calculate which layer of the shelf to place on based on how many objects are already there
            min_count = min(self.shelf_layer_occupied)
            layer_idx = next(
                idx for idx in reversed(range(len(self.shelf_layer_occupied)))
                if self.shelf_layer_occupied[idx] == min_count
            )
            pre_pose = PRE_SHELF_POSES[layer_idx]
            self.arm.move_pose(make_pose(*pre_pose))

            # change y position to align with the shelf layer
            place = list(pre_pose)
            place[1] = -0.30
        else:
            place = STORAGE_POSES.get(destination)

        self.arm.move_pose(make_pose(*place))
        
        move_gripper.gripper_open(self)

        if destination == 'shelf':
            self.shelf_layer_occupied[layer_idx] += 1
            self.arm.move_pose(make_pose(*pre_pose))
        
        return True

    def move_to_scout(self):
        self.get_logger().info('Moving to scout pose...')
        self.arm.move_pose(make_pose(*SCOUT_POSE))
        self.get_logger().info(f'Settling for {self.settle} seconds...')
        time.sleep(self.settle)

    def run_contest(self):
        raw = self.tasks
        tasks = raw['tasks'] if isinstance(raw, dict) and 'tasks' in raw else raw
        remaining = list(tasks)

        t0 = time.time()

        while remaining and (time.time() - t0) < self.time_budget:
            self.get_logger().info(f'=======Tasks remaining: {len(remaining)}=======')
            self.get_logger().info(f'Moving to home pose...')
                        
            names = sorted({t['object'].strip().lower() for t in remaining})
            dets = self.call_perception(names)
            if not dets:
                self.get_logger().warn('No detections from scout view; stopping. ')
                break

            pair = self.pick_target(dets, remaining)
            if pair is None:
                self.get_logger().warn('No detection matched a remaining task; stopping.')
                break
            
            det, task = pair
            self.get_logger().info(
                f"Target: {task['object']} -> {task['destination']} "
                f"(conf {det['conf']:.2f}, bbox {det['bbox']})")

            
            pose_opt = self.call_lift(det['bbox'])            
            obj_bl = self.transform_pose(pose_opt, self.optical_frame, self.base_frame)
            o = obj_bl.position
            # ------------- DEBUG: print the lifted pose in base_link, and compare to world_model truth if available -------------
            truth_world = self.get_object_truth(task['object'])     # [x,y,z,r,p,y] in world, or None
            truth_pose = make_pose(*truth_world)                # 6-list is (x,y,z,roll,pitch,yaw)
            truth_bl = self.transform_pose(truth_pose, 'world', self.base_frame)
            t = truth_bl.position

            self.get_logger().warn(f"lifted (base_link): x={o.x:.3f} y={o.y:.3f} z={o.z:.3f}")
            self.get_logger().warn(f"truth  (base_link): x={t.x:.3f} y={t.y:.3f} z={t.z:.3f}")
            self.get_logger().warn(f"error  (base_link): dx={o.x-t.x:+.3f} dy={o.y-t.y:+.3f} dz={o.z-t.z:+.3f}")
            # ------------------------------------------------------------------------------------------------------------------

            if self.grasp_and_place(pose_opt, task['destination'].strip().lower()):
                remaining.remove(task)


            self.get_logger().info("Moving to home pose...")
            self.arm.move_pose(make_pose(*HOME_POSE))

        elapsed = time.time() - t0
        self.get_logger().info(f'Contest loop ended after {elapsed:.1f}s with{len(remaining)} task(s) left.')


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