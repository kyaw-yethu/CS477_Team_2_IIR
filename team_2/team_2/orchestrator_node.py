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

from team_2_interfaces.srv import DetectObjects, PlanMotion
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


# --- SEAM 1/2/3: fill with real values ---------------------------------------
#                x     y     z     roll      pitch  yaw
HOME_POSE  = (0.0, 0.0, 0.6, math.pi,   0.0,   0.0)   # oblique view, tune
SCOUT_POSE   = (-0.20, 0.0, 0.55, math.pi,   -1.0,   0.0)
PLACE_POSES = {
    'left_storage':  (0.40,  0.30, 0.30, math.pi, 0.0, 0.0),
    'right_storage': (0.40, -0.30, 0.30, math.pi, 0.0, 0.0),
    'shelf':         (0.45,  0.00, 0.50, math.pi, 0.0, 0.0),
}
# -----------------------------------------------------------------------------


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter('camera_optical_frame',
                               'wrist_camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('settle_sec', 0.8)   # let the camera stream catch
        #                                              up to the new arm pose
        # NOTE: protocol PDF says 5 min, README says 8 min. Confirm with the TA.
        self.declare_parameter('time_budget_sec', 300.0)
        self.declare_parameter('motion_plan_service', 'plan_motion')
        self.declare_parameter('use_internal_object_pose_debug', False)
        self.declare_parameter('internal_pose_service', 'get_object_pose')
        self.declare_parameter('internal_pose_frame', 'base_link')
        self.declare_parameter('debug_autostart_tasks_json', '')

        self.optical_frame = self.get_parameter('camera_optical_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.settle = float(self.get_parameter('settle_sec').value)
        self.time_budget = float(self.get_parameter('time_budget_sec').value)
        self.use_internal_pose_debug = bool(
            self.get_parameter('use_internal_object_pose_debug').value
        )
        self.internal_pose_frame = self.get_parameter('internal_pose_frame').value
        self.debug_autostart_tasks_json = self.get_parameter(
            'debug_autostart_tasks_json'
        ).value

        self.tasks = None  # set when /parsed_tasks arrives -> leaves standby

        # Arm owner.
        self.arm = mj.ArmClient()

        # TF (camera optical -> base_link), populated while we spin.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service clients.
        self.detect_cli = self.create_client(
            DetectObjects, self.get_parameter('detect_service').value)
        self.lift_cli = self.create_client(
            StringPose, self.get_parameter('lift_service').value)
        self.plan_cli = self.create_client(
            PlanMotion, self.get_parameter('motion_plan_service').value)
        self.internal_pose_cli = self.create_client(
            StringPose, self.get_parameter('internal_pose_service').value)

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

        if self.use_internal_pose_debug and self.debug_autostart_tasks_json:
            try:
                parsed = json.loads(self.debug_autostart_tasks_json)
                self.tasks = parsed
                self.get_logger().warn(
                    '[DEBUG] Auto-start tasks loaded from debug_autostart_tasks_json.'
                )
            except json.JSONDecodeError as e:
                self.get_logger().error(
                    f'[DEBUG] Invalid debug_autostart_tasks_json: {e}'
                )

        self.get_logger().info(
            f'orchestrator_node: created. '
            f'debug_internal_pose={self.use_internal_pose_debug}'
        )

    # ---- standby trigger ----------------------------------------------------
    def on_tasks(self, msg):
        try:
            self.tasks = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Bad /parsed_tasks JSON: {e}')

    # ---- service helpers ----------------------------------------------------
    def wait_for_services(self):
        required = [
            (self.plan_cli, self.get_parameter('motion_plan_service').value),
        ]

        if self.use_internal_pose_debug:
            required.append(
                (self.internal_pose_cli, self.get_parameter('internal_pose_service').value)
            )
        else:
            required.extend([
                (self.detect_cli, 'detect_objects'),
                (self.lift_cli, 'lift_bbox_to_3d'),
            ])

        for cli, name in required:
            while rclpy.ok() and not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for '{name}' service...")

    def call_plan_motion(self, goal_pose, execute=True):
        req = PlanMotion.Request()
        req.goal_pose = goal_pose
        req.execute = bool(execute)
        fut = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error('plan_motion call failed (no response).')
            return False
        return bool(res.success)

    def call_internal_object_pose(self, object_name):
        req = StringPose.Request()
        req.data = object_name
        fut = self.internal_pose_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error('get_object_pose call failed (no response).')
            return None
        return res.pose

    def call_perception(self, names):
        req = DetectObjects.Request()
        req.data = json.dumps(names)
        self.get_logger().info(f'Calling perception for objects: {names}')
        fut = self.detect_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error('detect_objects call failed (no response).')
            return []
        try:
            detections = json.loads(res.data).get('detections', [])
            self.get_logger().info(f'Perception returned {len(detections)} detections.')
            return detections
        except json.JSONDecodeError:
            self.get_logger().error('detect_objects returned invalid JSON.')
            return []

    def call_lift(self, bbox):
        """bbox = [x1,y1,x2,y2] pixels -> 3D Pose in the camera optical frame,
        or None if the grasp server found no valid depth in the box."""
        req = StringPose.Request()
        req.data = json.dumps(bbox)
        self.get_logger().info(f'Calling lift service for bbox: {bbox}')
        fut = self.lift_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error('lift_bbox_to_3d call failed (no response).')
            return None
        p = res.pose
        # The grasp server returns an empty Pose (all zeros) on failure.
        if p.position.x == 0.0 and p.position.y == 0.0 and p.position.z == 0.0:
            self.get_logger().warn('lift_bbox_to_3d returned empty pose.')
            return None
        self.get_logger().info(
            f'Lift returned optical-frame pose: x={p.position.x:.3f}, '
            f'y={p.position.y:.3f}, z={p.position.z:.3f}'
        )
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

    # ---- arm motions (sole owner) -------------------------------------------
    def goto_scout(self):
        if not self.call_plan_motion(make_pose(*SCOUT_POSE), execute=True):
            self.get_logger().warn('Scout motion planning failed.')
        time.sleep(self.settle)

    def _plan_pose_or_fail(self, pose, label):
        if not self.call_plan_motion(pose, execute=True):
            self.get_logger().error(f'Planning failed for step: {label}')
            return False
        return True

    def normalize_destination(self, destination):
        d = str(destination).strip().lower()
        return d.replace(' ', '_')

    def grasp_and_place_base(self, obj_pose_bl, destination):
        """Pick/place from object pose in base frame using motion_planner."""
        move_gripper.gripper_open(self)

        pose_bl_to_gripper = self.arm.fk_request(
            [0.0, -np.pi / 2.0, 1.0, -np.pi / 3.0, -np.pi / 2.0, 0.0])

        goal = copy.copy(obj_pose_bl)
        goal.orientation = pose_bl_to_gripper.orientation

        # 1. approach from above
        goal.position.y -= 0.01
        goal.position.z += 0.10
        if not self._plan_pose_or_fail(goal, 'approach'):
            return False

        # 2. descend onto the object
        goal.position.z -= 0.11
        if not self._plan_pose_or_fail(goal, 'descend'):
            return False

        # 3. grasp
        move_gripper.gripper_close(self, force=0.5, gripper_close_pos=0.5)

        # 4. lift clear
        goal.position.z += 0.20
        if not self._plan_pose_or_fail(goal, 'lift'):
            return False

        # 5. carry to storage and release
        norm_dest = self.normalize_destination(destination)
        place = PLACE_POSES.get(norm_dest)
        if place is None:
            self.get_logger().warn(
                f"No place pose for destination '{destination}' (normalized '{norm_dest}'); releasing in place."
            )
        else:
            if not self._plan_pose_or_fail(make_pose(*place), 'place'):
                return False

        move_gripper.gripper_open(self)
        return True

    def grasp_and_place(self, obj_pose_optical, destination):
        """Adapted from example5 ObjectGraspClient.request_pick, plus a
        destination-aware place."""
        move_gripper.gripper_open(self)

        # Borrow a downward gripper orientation from FK of a known-good config.
        pose_bl_to_gripper = self.arm.fk_request(
            [0.0, -np.pi / 2.0, 1.0, -np.pi / 3.0, -np.pi / 2.0, 0.0])

        if not self.wait_for_tf():
            self.get_logger().error('TF never became available.')
            return False
        obj_bl = self.transform_pose(
            obj_pose_optical, self.optical_frame, self.base_frame)
        if obj_bl is None:
            return False

        return self.grasp_and_place_base(obj_bl, destination)

    # ---- main loop ----------------------------------------------------------
    # def move_home_pose(self):
    #     self.get_logger().info('Moving to home pose....')
    #     self.arm.move_pose(make_pose(*HOME_POSE))

    def run_contest(self):
        raw = self.tasks
        tasks = raw['tasks'] if isinstance(raw, dict) and 'tasks' in raw else raw
        remaining = list(tasks)
        self.get_logger().info(
            f'Leaving STANDBY. {len(remaining)} task(s): {remaining}')

        self.call_plan_motion(make_pose(*HOME_POSE), execute=True)
        t0 = time.time()

        while remaining and (time.time() - t0) < self.time_budget:
            if self.use_internal_pose_debug:
                task = remaining[0]
                self.get_logger().info(
                    f"[DEBUG] Internal pose mode: target {task['object']} -> {task['destination']}"
                )

                pose_internal = self.call_internal_object_pose(task['object'])
                if pose_internal is None:
                    self.get_logger().warn('[DEBUG] get_object_pose failed; stopping.')
                    break

                obj_bl = pose_internal
                if self.internal_pose_frame != self.base_frame:
                    obj_bl = self.transform_pose(
                        pose_internal,
                        self.internal_pose_frame,
                        self.base_frame,
                    )
                    if obj_bl is None:
                        self.get_logger().warn('[DEBUG] internal pose TF failed; re-scouting.')
                        continue

                if self.grasp_and_place_base(obj_bl, task['destination']):
                    remaining.remove(task)
                continue

            self.goto_scout()

            names = sorted({t['object'].strip().lower() for t in remaining})
            dets = self.call_perception(names)
            if not dets:
                self.get_logger().warn(
                    'No detections from scout view; stopping. '
                    '(Add an alternate scout pose here to retry.)')
                break

            pair = self.pick_target(dets, remaining)
            if pair is None:
                self.get_logger().warn(
                    'No detection matched a remaining task; stopping.')
                break
            det, task = pair
            self.get_logger().info(
                f"Target: {task['object']} -> {task['destination']} "
                f"(conf {det['conf']:.2f}, bbox {det['bbox']})")

            pose_opt = self.call_lift(det['bbox'])
            if pose_opt is None:
                # No depth in this box (often a bad/edge detection). Skip this
                # detection and re-scout rather than risk a blind grasp.
                self.get_logger().warn('Lift-to-3D failed; re-scouting.')
                continue

            if self.grasp_and_place(pose_opt, task['destination']):
                remaining.remove(task)

        elapsed = time.time() - t0
        self.get_logger().info(
            f'Contest loop ended after {elapsed:.1f}s. '
            f'{len(remaining)} task(s) left.')


def main():
    rclpy.init()
    node = Orchestrator()
    node.wait_for_services()

    # node.move_home_pose()

    node.get_logger().info('STANDBY: waiting for /parsed_tasks ...')
    while rclpy.ok() and node.tasks is None:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.tasks is not None:
        node.run_contest()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()