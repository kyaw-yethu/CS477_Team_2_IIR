#!/usr/bin/env python3
"""
compare_camera_coords.py

Utility script to compare object detections and coordinates between 
wrist and scene cameras. Useful for validating dual-camera perception accuracy.

Usage:
    ros2 run team_2 compare_camera_coords.py
"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from team_2_interfaces.srv import DetectObjects


class CameraComparator(Node):
    def __init__(self):
        super().__init__('camera_comparator')
        
        self.detect_cli = self.create_client(DetectObjects, 'detect_objects')
        
        # Wait for service
        while not self.detect_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for detect_objects service...")
        
        self.get_logger().info("Service ready. Testing both cameras...")
        
    def compare(self, object_names):
        """Detect same objects on both cameras and compare results."""
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Comparing detection for: {object_names}")
        self.get_logger().info(f"{'='*60}")
        
        dets_wrist, payload_wrist = self._detect_camera(object_names, 'wrist')
        dets_scene, payload_scene = self._detect_camera(object_names, 'scene')
        
        # Display results
        self.get_logger().info(f"\n[WRIST] Detected {len(dets_wrist)} objects:")
        for det in dets_wrist:
            bbox = det['bbox']
            self.get_logger().info(
                f"  {det['object']:12s} | bbox:[{bbox[0]:6.1f}, {bbox[1]:6.1f}, {bbox[2]:6.1f}, {bbox[3]:6.1f}] | conf:{det['conf']:.2f}")
        if not dets_wrist:
            self.get_logger().warn(f"Wrist response payload: {payload_wrist}")
        
        self.get_logger().info(f"\n[SCENE] Detected {len(dets_scene)} objects:")
        for det in dets_scene:
            bbox = det['bbox']
            self.get_logger().info(
                f"  {det['object']:12s} | bbox:[{bbox[0]:6.1f}, {bbox[1]:6.1f}, {bbox[2]:6.1f}, {bbox[3]:6.1f}] | conf:{det['conf']:.2f}")
        if not dets_scene:
            self.get_logger().warn(f"Scene response payload: {payload_scene}")
        
        # Compare detections
        self.get_logger().info(f"\n[COMPARISON]")
        wrist_objs = {d['object'].lower(): d for d in dets_wrist}
        scene_objs = {d['object'].lower(): d for d in dets_scene}
        
        all_objs = set(wrist_objs.keys()) | set(scene_objs.keys())
        
        for obj in sorted(all_objs):
            w_det = wrist_objs.get(obj)
            s_det = scene_objs.get(obj)
            
            if w_det and s_det:
                w_bbox = w_det['bbox']
                s_bbox = s_det['bbox']
                
                # Calculate center points
                w_cx = (w_bbox[0] + w_bbox[2]) / 2.0
                w_cy = (w_bbox[1] + w_bbox[3]) / 2.0
                s_cx = (s_bbox[0] + s_bbox[2]) / 2.0
                s_cy = (s_bbox[1] + s_bbox[3]) / 2.0
                
                # Calculate bbox sizes
                w_w = w_bbox[2] - w_bbox[0]
                w_h = w_bbox[3] - w_bbox[1]
                s_w = s_bbox[2] - s_bbox[0]
                s_h = s_bbox[3] - s_bbox[1]
                
                self.get_logger().info(
                    f"✓ {obj:12s} | Both detected | "
                    f"Wrist center:({w_cx:6.1f}, {w_cy:6.1f}) size:({w_w:5.1f}x{w_h:5.1f}) | "
                    f"Scene center:({s_cx:6.1f}, {s_cy:6.1f}) size:({s_w:5.1f}x{s_h:5.1f})")
            elif w_det:
                self.get_logger().warn(f"⚠ {obj:12s} | Wrist only (missed by scene cam)")
            elif s_det:
                self.get_logger().warn(f"⚠ {obj:12s} | Scene only (missed by wrist cam)")
        
        self.get_logger().info(f"{'='*60}\n")

    def _detect_camera(self, object_names, camera):
        req = DetectObjects.Request()
        payload = {'classes': object_names, 'camera': camera}
        req.data = json.dumps(payload)
        self.get_logger().info(f"Requesting {camera} camera: {payload}")

        fut = self.detect_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error(f"No response from detect_objects for {camera} camera")
            return [], '{}'

        if not res.data:
            self.get_logger().warn(f"Empty response.data from detect_objects for {camera} camera")
            return [], '{}'

        try:
            raw = res.data
            data = json.loads(raw)
            detections = data.get('detections', [])
            return detections, raw
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse {camera} response JSON: {e}")
            return [], res.data


def main():
    rclpy.init()
    node = CameraComparator()
    
    # Test with common objects
    test_objects = [
        'strawberry', 'book', 'soap', 'mustard', 'coke can', 'eraser'
    ]
    
    node.compare(test_objects)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
