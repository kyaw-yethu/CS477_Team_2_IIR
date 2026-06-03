#!/usr/bin/env python3
"""
perception_node.py  (Gemini, two-camera variant)

STANDBY perception SERVICE. Caches the latest RGB frame from BOTH the fixed
scene camera and the wrist camera, and exposes Gemini open-vocabulary detection
as a service that runs on EITHER camera, selected per request:

    team_2_interfaces/srv/DetectObjects   (service name: 'detect_objects')

      request.data  -- JSON, one of:
          ["coke can", "banana"]                      # bare list -> default cam
          {"classes": ["coke can"], "camera": "scene"}
          {"tasks":   [{"object": "coke can", ...}], "camera": "wrist"}

      response.data -- JSON:
          {"camera": "scene",
           "frame":  "camera_color_optical_frame",     # optical frame of that cam
           "detections": [{object, bbox, center, conf}, ...]}

WHY TWO CAMERAS (the geometry that forces this)
  A bbox found in the SCENE image indexes SCENE pixels; it cannot be lifted to
  3D with the WRIST depth (different sensor / FOV / pixels). So the competition
  flow is two-stage:

    1. scene detect  (arm clear, wide stable view)  -> WHAT is on the table and
       roughly where. Lift those boxes with the SCENE point cloud to get coarse
       base_link positions to drive the arm toward.
    2. wrist detect  (arm moved close, oblique)     -> precise box on the target,
       lifted with the WRIST point cloud for the actual grasp.

  This node serves both; the caller says which camera via the "camera" field.
  The response echoes back the camera AND its optical frame, so the lift service
  knows which point cloud to use and the orchestrator knows which frame to TF
  from. ONE detect call == ONE camera.

  >>> The matching lift service (grasp_server / 'lift_bbox_to_3d') must ALSO be
      made camera-aware: lift a 'scene' bbox against the scene cloud, a 'wrist'
      bbox against the wrist cloud. A scene bbox + wrist cloud silently returns
      garbage 3D. <<<

RUNTIME REQUIREMENT: needs INTERNET + a Gemini API key (same caveat as the
single-camera Gemini node). Pass it in:
    sudo docker run ... -e GEMINI_API_KEY="..." image_team_2
or set the 'api_key' ROS param.

Gemini box convention (AI Studio docs, same as example4/5): boxes come back as
[ymin, xmin, ymax, xmax] normalized 0-1000. If overlays look transposed, that's
the first thing to flip in _parse().
"""
import os
import re
import json
from functools import partial
from datetime import datetime

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from google import genai

from team_2_interfaces.srv import DetectObjects


# Matches one [ymin, xmin, ymax, xmax, 'label'] entry anywhere in the text
# (tolerates code fences / stray prose, and missing label quotes).
_BOX_RE = re.compile(
    r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
    r'\s*["\']?\s*([\w\s\-]+?)\s*["\']?\s*\]'
)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # --- API / model ----------------------------------------------------
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            api_key = self.get_parameter('api_key').get_parameter_value().string_value
        
        self.declare_parameter('api_key', api_key)
        self.declare_parameter('model', 'gemini-3-flash-preview')

        # --- cameras --------------------------------------------------------
        # Topic + optical frame for each camera. The optical frame is what the
        # lift service / orchestrator must TF from for that camera's detections.
        self.declare_parameter('scene_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('wrist_image_topic', '/wrist_camera/wrist_camera/color/image_raw')
        self.declare_parameter('scene_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('wrist_optical_frame', 'wrist_camera_color_optical_frame')
        self.declare_parameter('default_camera', 'wrist')

        # --- service / debug ------------------------------------------------
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('detections_topic', '/detections')
        # Drop whole-scene / whole-arm garbage boxes (>= 1.0 disables).
        self.declare_parameter('max_area_frac', 0.15)
        self.declare_parameter('save_annotated', True)

        api_key = os.getenv('GEMINI_API_KEY') \
            or self.get_parameter('api_key').get_parameter_value().string_value
        if not api_key:
            raise ValueError(
                'Gemini API key is not set. Set the GEMINI_API_KEY environment '
                "variable or provide the 'api_key' ROS parameter.")

        self.model = self.get_parameter('model').value
        self.client = genai.Client(api_key=api_key)
        
        # Fallback models if primary is unavailable
        self.fallback_models = [
            'gemini-2-5-flash',
            'gemini-2-5-flash-lite',
            'gemini-1-5-flash',
        ]

        self.default_camera = self.get_parameter('default_camera').value
        self.max_area_frac = float(self.get_parameter('max_area_frac').value)

        # Per-camera optical frames, reported back in every response.
        self.optical_frames = {
            'scene': self.get_parameter('scene_optical_frame').value,
            'wrist': self.get_parameter('wrist_optical_frame').value,
        }

        self.get_logger().info(
            f'Gemini two-camera backend, model={self.model}, '
            f'default_camera={self.default_camera}')

        self.bridge = CvBridge()
        # Latest BGR frame per camera.
        self.latest = {'scene': None, 'wrist': None}

        # Camera publishers are BEST_EFFORT -- subscriber MUST match or no frames
        # arrive (silent QoS-incompatibility, same gotcha as example4/5).
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, self.get_parameter('scene_image_topic').value,
            partial(self.on_image, cam='scene'), img_qos)
        self.create_subscription(
            Image, self.get_parameter('wrist_image_topic').value,
            partial(self.on_image, cam='wrist'), img_qos)

        # Debug publisher: still emit /detections for rqt. Orchestrator does NOT
        # read this -- it gets the same JSON back in the service response.
        self.pub = self.create_publisher(
            String, self.get_parameter('detections_topic').value, 10)

        self.srv = self.create_service(
            DetectObjects, self.get_parameter('detect_service').value,
            self.on_detect_request)

        self.get_logger().info(
            'perception_node: STANDBY (serving '
            f"'{self.get_parameter('detect_service').value}', cameras: "
            'scene + wrist)')

    def on_image(self, msg, cam):
        try:
            self.latest[cam] = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error ({cam}): {e}')

    def on_detect_request(self, request, response, save_annotated=True):
        """Reads the CURRENT frame from the requested camera and detects the
        requested classes on it."""
        try:
            parsed = json.loads(request.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Bad request JSON: {e}')
            return self._empty(response, self.default_camera)

        # --- which camera ---------------------------------------------------
        camera = self.default_camera
        if isinstance(parsed, dict):
            camera = str(parsed.get('camera', self.default_camera)).lower()
        if camera not in self.latest:
            self.get_logger().error(
                f"Unknown camera '{camera}'; valid: {list(self.latest)}.")
            return self._empty(response, self.default_camera)

        # --- which classes --------------------------------------------------
        if isinstance(parsed, dict):
            if 'classes' in parsed:
                src = parsed['classes']
                classes = sorted({str(c).strip().lower() for c in src if c})
            else:
                classes = sorted({t['object'].strip().lower()
                                  for t in parsed.get('tasks', [])
                                  if t.get('object')})
        else:
            classes = sorted({str(c).strip().lower() for c in parsed if c})

        if not classes:
            self.get_logger().warn('No target classes in request.')
            return self._empty(response, camera)

        frame = self.latest[camera].copy()

        self.get_logger().info("--------------------------------")
        self.get_logger().info(f"[{camera}] detecting classes: {classes}")

        try:
            detections = self.detect(frame, classes)
        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')
            return self._empty(response, camera)

        payload = json.dumps({
            'camera': camera,
            'frame': self.optical_frames[camera],
            'detections': detections,
        })
        response.data = payload

        out = String()
        out.data = payload
        self.pub.publish(out)

        if save_annotated:
            try:
                path = self.save_overlay(frame, detections, camera)
                self.get_logger().info(f'Saved annotated frame -> {path}')
            except Exception as e:
                self.get_logger().error(f'Could not save annotated frame: {e}')

        self.get_logger().info("--------------------------------")
        return response

    def _empty(self, response, camera):
        response.data = json.dumps({
            'camera': camera,
            'frame': self.optical_frames.get(camera, ''),
            'detections': [],
        })
        return response

    def detect(self, bgr, classes):
        """One Gemini call detects every instance of every requested class on
        the given frame. Includes fallback to alternative models if primary fails."""
        rgb = bgr[:, :, ::-1]                       # BGR (cv_bridge) -> RGB
        pil = PILImage.fromarray(np.ascontiguousarray(rgb))
        h, w = bgr.shape[:2]

        prompt = self._build_prompt(classes)
        
        # Try primary model first, then fallbacks
        models_to_try = [self.model] + self.fallback_models
        last_error = None
        
        for model in models_to_try:
            try:
                self.get_logger().info(f'Attempting detection with model: {model}')
                resp = self.client.models.generate_content(model=model, contents=[pil, prompt])
                text = resp.text or ''
                
                detections = self._parse(text, classes, w, h)
                detections = self._filter_oversize(detections, w, h)
                
                if model != self.model:
                    self.get_logger().warn(f'Primary model unavailable; used fallback: {model}')
                
                return detections
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                # Skip quota errors (429) - they'll retry later, don't waste fallbacks
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    self.get_logger().error(f'Model {model} quota exhausted (429). All models hit quota limit.')
                    raise RuntimeError(f'All models have exhausted free tier quota. Error: {e}')
                # Skip unavailable models (404) - try next one
                elif '404' in error_str or 'NOT_FOUND' in error_str:
                    self.get_logger().warn(f'Model {model} not found (404): trying next fallback...')
                    continue
                else:
                    self.get_logger().warn(f'Model {model} failed: {e}')
                    continue
        
        # All models failed
        if last_error:
            raise last_error
        raise RuntimeError('No models available for detection')

    def _build_prompt(self, classes):
        names = ', '.join(classes)
        return (
            'Detect every instance of the following objects in the image: '
            f'{names}. There can be one or several of each type. '
            'For EACH detected instance, output one list of the form '
            "[ymin, xmin, ymax, xmax, 'label'] using integer normalized "
            'coordinates in the range 0-1000, where label is copied EXACTLY '
            'from the object list above. Output only the lists, one per line, '
            'and no other text.'
        )

    def _parse(self, text, classes, w, h):
        detections = []
        for m in _BOX_RE.finditer(text):
            ymin, xmin, ymax, xmax = (int(m.group(i)) for i in (1, 2, 3, 4))
            label = self._match_label(m.group(5).strip().lower(), classes)
            if label is None:
                continue
            x1 = min(xmin, xmax) * w / 1000.0
            x2 = max(xmin, xmax) * w / 1000.0
            y1 = min(ymin, ymax) * h / 1000.0
            y2 = max(ymin, ymax) * h / 1000.0
            detections.append({
                'object': label,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                'conf': 1.0,                          # Gemini gives no score
            })
        return detections

    @staticmethod
    def _match_label(raw, classes):
        if raw in classes:
            return raw
        for c in classes:
            if c in raw or raw in c:
                return c
        raw_toks = set(raw.split())
        for c in classes:
            if raw_toks & set(c.split()):
                return c
        return None

    def _filter_oversize(self, detections, w, h):
        if self.max_area_frac >= 1.0:
            return detections
        img_area = float(w * h)
        kept = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            if (x2 - x1) * (y2 - y1) <= self.max_area_frac * img_area:
                kept.append(d)
            else:
                self.get_logger().warn(
                    f"Dropping oversize box for '{d['object']}' "
                    f'({(x2-x1)*(y2-y1)/img_area:.0%} of frame)')
        return kept

    def save_overlay(self, bgr, detections, camera, annotated_dir='/ros2_ws/src/team_2/detections'):
        rgb = bgr[:, :, ::-1]
        img = PILImage.fromarray(np.ascontiguousarray(rgb))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            cx, cy = d['center']
            tag = f"{d['object']} {d['conf']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
            if font is not None:
                l, t, r, b = draw.textbbox((0, 0), tag, font=font)
                tw, th = r - l, b - t
            else:
                tw, th = 8 * len(tag), 12
            ty = max(0, y1 - th - 4)
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=(0, 255, 0))
            draw.text((x1 + 3, ty + 2), tag, fill=(0, 0, 0), font=font)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        os.makedirs(annotated_dir, exist_ok=True)
        path = os.path.join(annotated_dir, f'det_{camera}_{stamp}.png')
        img.save(path)
        return path


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()