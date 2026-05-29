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


class GraspServerNode(Node):
    def __init__(self):
        super().__init__('grasp_server')

        self.declare_parameter('lift_service', 'lift_bbox_to_3d')
        self.declare_parameter(
            'points_topic',
            '/wrist_camera/wrist_camera/depth/color/points')
        self.declare_parameter('optical_frame',
                               'wrist_camera_color_optical_frame')
        # How far below the topmost point (metres) to average for the grasp
        # centre — same 2 cm slab as example5.
        self.declare_parameter('top_slab_m', 0.02)

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
        self.publisher = self.create_publisher(
            PointCloud2, '/roi_filtered_points', roi_qos)

        self.srv = self.create_service(
            StringPose, self.get_parameter('lift_service').value,
            self.lift_callback)

        self.latest_cloud = None
        self.get_logger().info(
            'grasp_server: STANDBY (serving '
            f"'{self.get_parameter('lift_service').value}')")

    def points_callback(self, msg):
        """Point Cloud subscriber callback (unchanged from example5)."""
        raw_cloud = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        # Organized cloud (height > 1) -> (H, W, 3); else keep 1D.
        if msg.height > 1:
            self.latest_cloud = raw_cloud.reshape(msg.height, msg.width, 3)
        else:
            self.latest_cloud = raw_cloud

    def lift_callback(self, request, response):
        """bbox JSON [x1,y1,x2,y2] -> 3D centre Pose in the optical frame."""
        if self.latest_cloud is None:
            self.get_logger().warn('No point cloud received yet.')
            response.pose = Pose()
            return response

        try:
            x1, y1, x2, y2 = [int(round(float(v)))
                              for v in json.loads(request.data)]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.get_logger().error(f'Bad bbox in request.data: {e}')
            response.pose = Pose()
            return response

        # NOTE: get_bbox_center_3d expects (top, left, bottom, right) = (py1, px1, py2, px2).
        center_3d = self.get_bbox_center_3d(y1, x1, y2, x2)

        if center_3d is None:
            self.get_logger().info('No valid depth in bbox; returning empty Pose.')
            response.pose = Pose()        # all-zero -> orchestrator skips
            return response

        p = Pose()
        p.position.x = float(center_3d[0])
        p.position.y = float(center_3d[1])
        p.position.z = float(center_3d[2])
        response.pose = p
        self.get_logger().info(f'3D centre (optical frame): {center_3d}')
        return response

    # ---- unchanged from example5 -------------------------------------------
    def get_bbox_center_3d(self, py1, px1, py2, px2):
        """Compute the 3D center of a bounding box region using NumPy slicing."""
        if self.latest_cloud is None:
            self.get_logger().info("No cloud msg")
            return None

        # 1. Slice the region of interest (BBox), clipped to valid bounds.
        if self.latest_cloud.ndim != 3:
            self.get_logger().warn("Cloud is not organized (H,W,3); cannot slice bbox.")
            return None
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

        # 3. Keep points within `top_slab` of the topmost (min-Z = closest) point.
        z_values = valid_points[:, 2]
        z_min = np.min(z_values)
        z_threshold = z_min + self.top_slab

        top_mask = (z_values >= z_min) & (z_values <= z_threshold)
        target_points = valid_points[top_mask]

        if target_points.size == 0:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 4. Mean of the top slab.
        return np.mean(target_points, axis=0)

    def publish_roi_cloud(self, valid_points, frame_id=None):
        """Publish the ROI points as PointCloud2 (debug/RViz). Unchanged from
        example5 except the frame defaults to the configured optical frame."""
        if frame_id is None:
            frame_id = self.optical_frame
        if valid_points is None or len(valid_points) == 0:
            self.get_logger().warn("There are no valid points.")
            return

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        points_float32 = valid_points[:, :3].astype(np.float32)
        data = points_float32.tobytes()

        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points_float32),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,                       # float32 * 3
            row_step=12 * len(points_float32),
            data=data,
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