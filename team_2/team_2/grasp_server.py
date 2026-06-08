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
        self.declare_parameter('support_lift_m', 0.008)   # keep points standing >this above the table
        self.declare_parameter('plane_tol_m', 0.006)      # RANSAC inlier band for the support plane

        self.near_pct = float(self.get_parameter('near_surface_pct').value)
        self.support_lift = float(self.get_parameter('support_lift_m').value)
        self.plane_tol = float(self.get_parameter('plane_tol_m').value)
        self.plane_iters = 200

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

    @staticmethod
    def _grow(mask, region):
        """One step of 4-connected dilation of `region`, clipped to `mask`."""
        g = region.copy()
        g[1:, :]  |= region[:-1, :]
        g[:-1, :] |= region[1:, :]
        g[:, 1:]  |= region[:, :-1]
        g[:, :-1] |= region[:, 1:]
        return g & mask

    def _components(self, mask):
        """Label all 4-connected components of a bool mask. Returns a list of
        (component_mask, pixel_count, bbox_area). Pure numpy, no scipy."""
        remaining = mask.copy()
        comps = []
        while remaining.any():
            ys, xs = np.nonzero(remaining)
            region = np.zeros_like(mask)
            region[int(ys[0]), int(xs[0])] = True
            prev = -1
            while region.sum() != prev:                 # flood fill this component
                prev = int(region.sum())
                region = self._grow(remaining, region)
            ys, xs = np.nonzero(region)
            bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            comps.append((region, int(region.sum()), int(bbox_area)))
            remaining &= ~region
        return comps

    def _select_target_blob(self, slab2d):
        """Pick the slab component that is the detection's target. The bbox is
        tight around the target, so the target's own bbox spans the whole ROI
        while a neighbour only clips a corner/edge -- so choose the component
        whose bbox fills the most of the ROI, and among comparable ones the most
        central.

        This replaces a centre-pixel seed, which fails for a thin diagonal object
        (a hammer handle): its axis-aligned bbox centre sits in empty space and
        can land on a neighbour (the strawberry beside the head)."""
        comps = self._components(slab2d)
        if len(comps) <= 1:
            return comps[0][0] if comps else slab2d
        h, w = slab2d.shape
        cy, cx = h / 2.0, w / 2.0
        max_bbox = max(bbox for _, _, bbox in comps)
        cand = [c for c in comps if c[2] >= 0.7 * max_bbox and c[1] >= 20] or comps

        def off_center(comp):
            ys, xs = np.nonzero(comp[0])
            return (ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2

        return min(cand, key=off_center)[0]

    def _grasp_center(self, points, yaw):
        """On-object grasp point. np.mean() lands in the hollow of a curved object
        (banana), so take the thin cross-slice at the MIDDLE of the long axis, use
        that slice's centroid (centred along length and across width), then snap to
        the nearest real point. Convex objects are essentially unaffected."""
        xy = points[:, :2].astype(np.float64)
        u = np.array([np.cos(yaw), np.sin(yaw)])           # long-axis direction
        t = (xy - xy.mean(axis=0)) @ u                     # coord along long axis
        band = np.abs(t - np.median(t)) <= max(0.01, 0.10 * np.ptp(t))
        slice_pts = points[band] if band.any() else points
        center = slice_pts.mean(axis=0)
        k = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
        return points[k]                                   # snapped onto the object

    def _fit_support_plane(self, cloud):
        """RANSAC-fit the dominant plane (table/floor) over the WHOLE cloud, where
        it is unambiguously the largest flat surface. Returns (normal, point) with
        the normal oriented toward the optical origin, or None. Fitting on the full
        scene -- not the tight bbox -- is what makes this robust: a short object's
        top sits inside the depth slab next to the floor, and an in-bbox fit can
        mistake a flat object top for the support surface."""
        if cloud is None or cloud.ndim != 3:
            return None
        P = cloud.reshape(-1, 3)
        P = P[~np.isnan(P).any(axis=1)]
        if len(P) < 200:
            return None
        rng = np.random.default_rng(0)
        S = P if len(P) <= 4000 else P[rng.choice(len(P), 4000, replace=False)]
        best_inl, best = 0, None
        for _ in range(self.plane_iters):
            a, b, c = S[rng.choice(len(S), 3, replace=False)]
            nrm = np.cross(b - a, c - a)
            L = np.linalg.norm(nrm)
            if L < 1e-9:
                continue
            nrm /= L
            inl = int(np.count_nonzero(np.abs((S - a) @ nrm) < self.plane_tol))
            if inl > best_inl:
                best_inl, best = inl, (nrm.copy(), a.copy())
        if best is None or best_inl < 0.20 * len(S):
            return None
        nrm, a = best
        if nrm @ (-a) < 0.0:                    # point the normal at the camera
            nrm = -nrm
        return nrm, a
    
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

        fig.suptitle(f'grasp mean (optical) = ({mean_pt[0]:.3f}, {mean_pt[1]:.3f}, {mean_pt[2]:.3f}) m')
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
        """Lift a 2D bbox to a 3D grasp point + long-axis yaw, in the optical frame.

        Each step fixes one failure mode:
          2b. remove support plane (table/floor) -> short object vs the depth slab
          3.  near-surface depth slab            -> flying pixels
          3b. blob under the bbox centre         -> a neighbour clipped into the box
          4.  on-object centre + long axis       -> curved objects (banana)
        """
        if self.latest_cloud is None:
            self.get_logger().info("No cloud msg")
            return None

        # 1. Slice the ROI, clipped to valid bounds.
        h, w, _ = self.latest_cloud.shape
        py1, py2 = np.clip([py1, py2], 0, h - 1)
        px1, px2 = np.clip([px1, px2], 0, w - 1)
        roi_cloud = self.latest_cloud[py1:py2, px1:px2]

        # 2. Drop NaNs; keep the 2D mask for the connectivity step (3b).
        valid2d = ~np.isnan(roi_cloud).any(axis=2)
        valid_points = roi_cloud[valid2d]
        if len(valid_points) > 0:
            self.publish_roi_cloud(valid_points)
        else:
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 2b. Remove the support surface (table/floor). A short object's top sits
        #     less than `top_slab` above the floor, so the floor falls inside the
        #     depth band -- object-dependent (a taller banana clears it, a flatter
        #     one doesn't). Fit the plane on the FULL cloud (floor dominates there)
        #     and keep only what stands proud of it. No plane / empty -> keep all.
        plane = self._fit_support_plane(self.latest_cloud)
        if plane is not None:
            nrm, p0 = plane
            height = (roi_cloud - p0) @ nrm                # (h, w); NaN where invalid
            obj2d = valid2d & (height > self.support_lift)
            if int(obj2d.sum()) < 20:                      # gate emptied it -> keep all
                obj2d = valid2d
        else:
            obj2d = valid2d

        # 3. Robust near-surface slab on the de-floored points (low percentile, not
        #    raw min, so a flying pixel can't anchor the slab on garbage).
        z_grid = roi_cloud[:, :, 2]
        z_anchor = np.percentile(roi_cloud[obj2d][:, 2], self.near_pct)
        z_threshold = z_anchor + self.top_slab
        slab2d = obj2d & (z_grid >= z_anchor) & (z_grid <= z_threshold)
        if not slab2d.any():
            self.get_logger().info("No points!!!!!!!!!!!!!!!!")
            return None

        # 3b. Pick the component that is the detector's target (its bbox fills the ROI), not whatever happens to sit at the geometric centre.
        slab2d = self._select_target_blob(slab2d)
        target_points = roi_cloud[slab2d]

        # 4. Long axis (PCA) + an ON-OBJECT centre.
        yaw = self._principal_axis_yaw(target_points)
        mean_pt = self._grasp_center(target_points, yaw)
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