#!/usr/bin/env python3
"""
grasp_server.py

Lifts a 2D bounding box to a 3D grasp point. Refactor of example5_grasp_server:
Gemini/VLM detection is REMOVED (perception_node does that now) — this node only
does the depth/point-cloud geometry, which is the part worth keeping.

Service:  riro_srvs/srv/StringPose   (service name: 'lift_bbox_to_3d')
    request.data  = JSON bbox in PIXELS, "[x1, y1, x2, y2]"  (xyxy)
    response.pose = Pose, position = 3D centre in 'wrist_camera_color_optical_frame'
                    (metres). On failure (no valid depth in box) returns an
                    all-zero Pose; the orchestrator treats that as "skip/re-scout".

The orchestrator TF-transforms the returned pose optical_frame -> base_link
itself, so this node stays frame-agnostic — exactly like example5.

get_bbox_center_3d() and publish_roi_cloud() are unchanged from example5.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
import std_msgs.msg
from geometry_msgs.msg import Pose

from riro_srvs.srv import StringPose

import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')          # headless: no display needed inside the container
import matplotlib.pyplot as plt

class GraspServerNode(Node):
    def __init__(self):
        super().__init__('grasp_server')

        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter('points_topic', '/wrist_camera/wrist_camera/depth/color/points')
        self.declare_parameter('optical_frame', 'wrist_camera_color_optical_frame')
        # How far below the topmost point (metres) to average for the grasp centre (default: 2 cm)
        self.declare_parameter('top_slab_m', 0.05)

        self.optical_frame = self.get_parameter('optical_frame').value
        self.top_slab = float(self.get_parameter('top_slab_m').value)

        # Camera publishers are BEST_EFFORT — subscriber MUST match or no data.
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.subscription_pts = self.create_subscription(
            PointCloud2,
            self.get_parameter('points_topic').value,
            self.points_callback,
            qos_profile)

        # Latched debug publisher of the ROI cloud (view in RViz2).
        roi_qos = QoSProfile(depth=10)
        roi_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        roi_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(PointCloud2, '/roi_filtered_points', roi_qos)
        
        self.declare_parameter('near_surface_pct', 2.0)   # reject the closest N% as flying pixels
        self.near_pct = float(self.get_parameter('near_surface_pct').value)

        self.srv = self.create_service(StringPose, self.get_parameter('lift_service').value, self.lift_callback)

        self.latest_cloud = None
        self.get_logger().info(
            'grasp_server: STANDBY (serving '
            f"'{self.get_parameter('lift_service').value}')")

    def _principal_axis_yaw(self, points):
        """Long-axis angle (rad, about optical +Z) of the points' XY footprint.
        PCA: principal eigenvector of the 2D covariance. The 180° sign ambiguity of
        the eigenvector doesn't matter — the orchestrator folds the angle into
        [0, pi) anyway (jaws are symmetric under 180°)."""
        xy = points[:, :2].astype(np.float64)
        if len(xy) < 3:
            return 0.0
        xy = xy - xy.mean(axis=0)
        cov = np.cov(xy, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return 0.0
        evals, evecs = np.linalg.eigh(cov)        # ascending
        major = evecs[:, int(np.argmax(evals))]   # principal direction
        return float(np.arctan2(major[1], major[0]))

    def save_xy_plot(self, valid_points, target_points, mean_pt, yaw=None,
                 out_dir='/ros2_ws/src/team_2/debug'):
        """Top-down + image-plane scatter of the ROI cloud and the grasp mean.

        Frame = wrist_camera_color_optical_frame (x=right, y=down, z=depth).
        Left panel (X-Z) is the bird's-eye 'reaching' view; the red X is where
        the gripper is told to go. (In base_link that same plane is X-Y.)
        """
        os.makedirs(out_dir, exist_ok=True)
        fig, (ax_top, ax_img) = plt.subplots(1, 2, figsize=(12, 5))

        # bird's-eye: X (right) vs Z (depth) -- this reveals the surface bias
        ax_top.scatter(valid_points[:, 0],  valid_points[:, 2],  s=2, c='0.7', label='valid ROI')
        ax_top.scatter(target_points[:, 0], target_points[:, 2], s=4, c='tab:blue', label='top slab')
        ax_top.scatter(mean_pt[0], mean_pt[2], s=140, marker='X', c='red', zorder=5, label='grasp mean')
        ax_top.set_xlabel('x  (right, m)'); ax_top.set_ylabel('z  (depth, m)')
        ax_top.set_title('Top-down  X-Z'); ax_top.invert_yaxis()   # nearer = up
        ax_top.set_aspect('equal', 'box'); ax_top.grid(True, alpha=0.3); ax_top.legend(fontsize=8)

        # image plane: X (right) vs Y (down)
        ax_img.scatter(valid_points[:, 0],  valid_points[:, 1],  s=2, c='0.7')
        ax_img.scatter(target_points[:, 0], target_points[:, 1], s=4, c='tab:blue')
        ax_img.scatter(mean_pt[0], mean_pt[1], s=140, marker='X', c='red', zorder=5)
        ax_img.set_xlabel('x  (right, m)'); ax_img.set_ylabel('y  (down, m)')
        ax_img.set_title('Image plane  X-Y'); ax_img.invert_yaxis()
        ax_img.set_aspect('equal', 'box'); ax_img.grid(True, alpha=0.3)

        if yaw is not None:
            L = 0.05  # 5 cm half-length
            dx, dy = np.cos(yaw), np.sin(yaw)
            ax_img.plot([mean_pt[0] - L*dx, mean_pt[0] + L*dx],
                        [mean_pt[1] - L*dy, mean_pt[1] + L*dy],
                        c='tab:green', lw=2, label='long axis')
            ax_img.legend(fontsize=8)

        fig.suptitle(f'grasp mean (optical) = '
                    f'({mean_pt[0]:.3f}, {mean_pt[1]:.3f}, {mean_pt[2]:.3f}) m')
        fig.tight_layout()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(out_dir, f'grasp_xy_{stamp}.png')
        fig.savefig(path, dpi=120); plt.close(fig)
        # self.get_logger().info(f'Saved grasp plot -> {path}')
        return path

    def points_callback(self, msg):
        """Point Cloud subscriber callback"""
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))

        # 2. Reshape the data into (H, W, 3) using the message height and width.
        # For an organized point cloud, height > 1.
        if msg.height > 1:
            self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3)
        else:
            # For an unorganized cloud, keep the 1D layout or handle it separately.
            self.latest_cloud = raw_cloud    

    def lift_callback(self, request, response):
        """bbox JSON [x1,y1,x2,y2] -> 3D centre Pose in the optical frame."""
        if self.latest_cloud is None:
            self.get_logger().warn('No point cloud received yet.')
            response.pose = Pose()
            return response

        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in json.loads(request.data)]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.get_logger().error(f'Bad bbox in request.data: {e}')
            response.pose = Pose()
            return response

        # NOTE: get_bbox_center_3d expects (top, left, bottom, right) = (py1, px1, py2, px2).
        result = self.get_bbox_center_3d(y1, x1, y2, x2)
        if result is None:
            self.get_logger().info('No valid depth in bbox; returning empty Pose.')
            response.pose = Pose()        # all-zero -> orchestrator skips
            return response

        center_3d, yaw = result

        p = Pose()
        p.position.x = float(center_3d[0])
        p.position.y = float(center_3d[1])
        p.position.z = float(center_3d[2])
        # long axis packed as a rotation about optical +Z; orchestrator decodes
        # it with 2*atan2(q.z, q.w).
        p.orientation.z = float(np.sin(yaw / 2.0))
        p.orientation.w = float(np.cos(yaw / 2.0))
        response.pose = p
        self.get_logger().info(f'3D centre (optical): {center_3d}, long-axis yaw: {np.degrees(yaw):.0f} deg')
        return response

    # ---- From example5 -------------------------------------------
    def get_bbox_center_3d(self, py1, px1, py2, px2):
        """Compute the 3D center of a bounding box region using NumPy slicing."""
        if self.latest_cloud is None:
            self.get_logger().info("No cloud msg")
            return None

        # 1. Slice the region of interest (BBox), clipped to valid bounds.
        h, w, _ = self.latest_cloud.shape
        py1, py2 = np.clip([py1, py2], 0, h - 1)
        px1, px2 = np.clip([px1, px2], 0, w - 1)

        roi_cloud = self.latest_cloud[py1:py2, px1:px2]

        # 2. Remove NaN values (points with missing depth).
        mask = ~np.isnan(roi_cloud).any(axis=2)
        valid_points = roi_cloud[mask]

        if len(valid_points) > 0:
            self.publish_roi_cloud(valid_points)
        else:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 3. Robust near-surface slab.
        # A few near-camera outliers (flying pixels at depth-discontinuity edges,
        # or a neighbour clipped by the bbox) can sit several cm IN FRONT of the
        # object. np.min() then anchors the slab on that garbage and the mean
        # lands nowhere near the object -- exactly the banana case. Anchor on a
        # low percentile instead of the raw minimum.
        z_values = valid_points[:, 2]
        z_anchor = np.percentile(z_values, self.near_pct)   # robust 'nearest'
        z_threshold = z_anchor + self.top_slab

        top_mask = (z_values >= z_anchor) & (z_values <= z_threshold)
        target_points = valid_points[top_mask]

        if target_points.size == 0:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 4. Mean of the top slab.
        mean_pt = np.mean(target_points, axis=0)
        yaw = self._principal_axis_yaw(target_points)
        try:
            self.save_xy_plot(valid_points, target_points, mean_pt, yaw)
        except Exception as e:
            self.get_logger().warn(f'plot failed: {e}')
        return mean_pt, yaw

    def publish_roi_cloud(self, valid_points, frame_id='wrist_camera_color_optical_frame'):
        """
        valid_points: NumPy array with shape (N, 3) or (N, 4)
        frame_id: Frame name to use as the Fixed Frame in RViz2
        """
        if valid_points is None or len(valid_points) == 0:
            self.get_logger().warn("There are no valid points.")
            return

        # 1. Set the PointCloud2 message header
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        # 2. Define the fields for x, y, and z coordinates
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # 3. Convert the NumPy array into byte data
        # Convert to float32 if valid_points is float64.
        points_float32 = valid_points[:, :3].astype(np.float32)
        data = points_float32.tobytes()

        # 4. Assemble the message
        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points_float32),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,  # float32(4 bytes) * 3
            row_step=12 * len(points_float32),
            data=data
        )

        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = GraspServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()