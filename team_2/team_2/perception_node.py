#!/usr/bin/env python3
"""
perception_node.py  (Gemini variant)

STANDBY perception SERVICE. Caches the latest RGB frame from the wrist camera
and exposes Gemini open-vocabulary detection as a service:

    team_2_interfaces/srv/DetectObjects   (service name: 'detect_objects')
        request.data  = JSON array of class names, e.g. ["coke can","banana"]
                        (also accepts {"tasks":[{"object":...}, ...]})
        response.data = {"detections":[{object,bbox,center,conf}, ...]}

This is a DROP-IN replacement for the Grounding DINO version: the service name,
the request/response JSON shapes, the bbox=[x1,y1,x2,y2] pixel convention, the
/detections debug publish, and save_overlay() are all unchanged, so the
orchestrator never knows the backend changed.

WHAT CHANGED vs the Grounding DINO node
  - No torch / transformers / local weights. One network call per request to the
    Gemini API instead.
  - One call detects ALL requested classes and ALL their instances at once
    (cheaper + faster than per-class, and naturally handles "same type appears
    N times").
  - Gemini gives no per-box confidence, so conf is fixed at 1.0. The
    orchestrator's pick_target() then degenerates to "first matching detection"
    (its `det['conf'] > best['conf']` test is never true when all are equal) —
    fine here, since detection order from the model is arbitrary anyway.

>>> RUNTIME REQUIREMENT: this node needs INTERNET + a Gemini API key. <<<
    That breaks the "fully offline competition box" assumption the baked-in
    Grounding DINO + Qwen design was built around. Confirm the competition
    machine has network before relying on this. The instruction_parser (Qwen,
    local) stays offline-capable; only perception now needs the network.

    Pass the key into the container, e.g.:
        sudo docker run ... -e GEMINI_API_KEY="..." image_team_2
    or set the 'api_key' ROS param via params.yaml.

Gemini box convention (per AI Studio docs, same as example4/5): boxes come back
as [ymin, xmin, ymax, xmax] in normalized 0-1000 coords. If your overlays look
transposed, that's the first thing to flip.

Output contract (unchanged):
  {"detections": [{"object": str,
                   "bbox":   [x1, y1, x2, y2],   # pixels
                   "center": [cx, cy],           # pixels
                   "conf":   float}]}            # always 1.0 with Gemini
"""
import os
import re
import json
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
# (tolerates code fences / stray prose around it, and missing label quotes).
_BOX_RE = re.compile(
    r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
    r'\s*["\']?\s*([\w\s\-]+?)\s*["\']?\s*\]'
)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # --- API / model ----------------------------------------------------
        self.declare_parameter('api_key', '')
        # Optional offline/debug detections: JSON string to return instead of
        # calling an external API. See config/params.yaml debug_detections.
        self.declare_parameter('debug_detections', '')
        # example4 used 'gemini-2.5-flash' for detection; example5 used
        # 'gemini-3.1-flash-lite-preview'. Switch here via param if you like.
        self.declare_parameter('model', 'gemini-2.5-flash')

        # --- topics / service ----------------------------------------------
        self.declare_parameter(
            'image_topic', '/wrist_camera/wrist_camera/color/image_raw')
        self.declare_parameter('detect_service', 'detect_objects')
        self.declare_parameter('detections_topic', '/detections')

        # --- post-processing ------------------------------------------------
        # Drop whole-scene / whole-arm garbage boxes (set >= 1.0 to disable).
        # Tune to your largest real object's on-screen footprint.
        self.declare_parameter('max_area_frac', 0.15)

        # --- debug overlays -------------------------------------------------
        self.declare_parameter('save_annotated', True)
        self.declare_parameter('annotated_dir', '/models/detections')

        api_key = os.getenv('GEMINI_API_KEY') \
            or self.get_parameter('api_key').get_parameter_value().string_value
        self.model = self.get_parameter('model').value
        self.debug_detections = self.get_parameter('debug_detections').value

        if not api_key:
            self.get_logger().warn(
                'No Gemini API key set — using debug/local detection fallback.')
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

        self.max_area_frac = float(self.get_parameter('max_area_frac').value)

        self.save_annotated = bool(self.get_parameter('save_annotated').value)
        self.annotated_dir = self.get_parameter('annotated_dir').value
        if self.save_annotated:
            os.makedirs(self.annotated_dir, exist_ok=True)
            self.get_logger().info(f'Annotated frames -> {self.annotated_dir}')

        self.get_logger().info(f'Gemini detection backend, model={self.model}')

        self.bridge = CvBridge()
        self.latest = None

        # Camera publishers are BEST_EFFORT — the subscriber MUST match or no
        # frames arrive (silent QoS-incompatibility, same gotcha as example4/5).
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self.on_image, img_qos)

        # Debug publisher: still emit /detections so you can watch in rqt. The
        # orchestrator does NOT read this — it gets the same JSON in the response.
        self.pub = self.create_publisher(
            String, self.get_parameter('detections_topic').value, 10)

        # The trigger: a service instead of a topic callback.
        self.srv = self.create_service(
            DetectObjects, self.get_parameter('detect_service').value,
            self.on_detect_request)

        self.get_logger().info(
            'perception_node: STANDBY (serving '
            f"'{self.get_parameter('detect_service').value}')")

    def on_image(self, msg):
        try:
            self.latest = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    def on_detect_request(self, request, response):
        """Service handler. Reads the CURRENT frame (post-scout-move) and runs
        detection on the requested classes."""
        if self.latest is None:
            self.get_logger().warn('No camera frame yet; returning empty.')
            response.data = json.dumps({'detections': []})
            return response

        # Accept either a bare JSON array of names or a {"tasks":[...]} object.
        try:
            parsed = json.loads(request.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Bad request JSON: {e}')
            response.data = json.dumps({'detections': []})
            return response

        if isinstance(parsed, dict):
            classes = sorted({t['object'].strip().lower()
                              for t in parsed.get('tasks', [])
                              if t.get('object')})
        else:
            classes = sorted({str(c).strip().lower() for c in parsed if c})

        if not classes:
            self.get_logger().warn('No target classes in request.')
            response.data = json.dumps({'detections': []})
            return response

        self.get_logger().info(f'Detecting classes: {classes}')
        frame = self.latest.copy()
        try:
            if self.client is None:
                # Offline/debug mode: return provided debug detections if any.
                if self.debug_detections:
                    try:
                        dbg = json.loads(self.debug_detections)
                        detections = dbg.get('detections', [])
                        self.get_logger().info('Returning debug detections (offline mode).')
                    except Exception as e:
                        self.get_logger().error(f'Bad debug_detections JSON: {e}')
                        detections = []
                else:
                    self.get_logger().warn('No detection backend available; returning empty list.')
                    detections = []
            else:
                detections = self.detect(frame, classes)
        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')
            response.data = json.dumps({'detections': []})
            return response

        payload = json.dumps({'detections': detections})
        response.data = payload

        # Mirror to the debug topic + disk (unchanged behaviour).
        out = String()
        out.data = payload
        self.pub.publish(out)
        self.get_logger().info(
            f'Returned {len(detections)} detection(s): {payload}')

        if self.save_annotated:
            try:
                path = self.save_overlay(frame, detections)
                self.get_logger().info(f'Saved annotated frame -> {path}')
            except Exception as e:
                self.get_logger().error(f'Could not save annotated frame: {e}')

        return response

    def detect(self, bgr, classes):
        """One Gemini call detects every instance of every requested class.

        Querying all classes together (rather than per-class) is cheaper and
        faster, and because we instruct the model to emit one bracketed list per
        instance, multiple objects of the same type come back as separate
        detections for free.
        """
        # BGR (cv_bridge) -> RGB -> PIL, as the API expects.
        rgb = bgr[:, :, ::-1]
        pil = PILImage.fromarray(np.ascontiguousarray(rgb))
        h, w = bgr.shape[:2]

        prompt = self._build_prompt(classes)
        resp = self.client.models.generate_content(
            model=self.model, contents=[pil, prompt])
        text = resp.text or ''
        self.get_logger().info(f'Gemini raw: {text.strip()}')

        detections = self._parse(text, classes, w, h)
        detections = self._filter_oversize(detections, w, h)
        return detections

    def _build_prompt(self, classes):
        names = ', '.join(classes)
        return (
            'Detect every instance of the following objects in the image: '
            f'{names}. There may be zero, one, or several of each type. '
            'For EACH detected instance, output one list of the form '
            "[ymin, xmin, ymax, xmax, 'label'] using integer normalized "
            'coordinates in the range 0-1000, where label is copied EXACTLY '
            'from the object list above. Output only the lists, one per line, '
            'and no other text. If none of the objects are present, output '
            'nothing.'
        )

    def _parse(self, text, classes, w, h):
        """Pull every [ymin,xmin,ymax,xmax,label] out of the model text and map
        each label back onto one of the requested classes."""
        detections = []
        for m in _BOX_RE.finditer(text):
            ymin, xmin, ymax, xmax = (int(m.group(i)) for i in (1, 2, 3, 4))
            label = self._match_label(m.group(5).strip().lower(), classes)
            if label is None:
                # Gemini named something we didn't ask for — ignore it.
                continue

            # normalized 0-1000 -> pixels; min/max guards against swapped corners
            x1 = min(xmin, xmax) * w / 1000.0
            x2 = max(xmin, xmax) * w / 1000.0
            y1 = min(ymin, ymax) * h / 1000.0
            y2 = max(ymin, ymax) * h / 1000.0

            detections.append({
                'object': label,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                'conf': 1.0,                      # Gemini gives no score
            })
        return detections

    @staticmethod
    def _match_label(raw, classes):
        """Map a returned label to a requested class. Handles 'coke' -> 'coke
        can', 'meat' -> 'meat can', exact matches, and single-word overlaps."""
        if raw in classes:
            return raw
        for c in classes:                         # substring either direction
            if c in raw or raw in c:
                return c
        raw_toks = set(raw.split())               # any shared word
        for c in classes:
            if raw_toks & set(c.split()):
                return c
        return None

    def _filter_oversize(self, detections, w, h):
        """Drop the whole-scene / whole-arm garbage boxes Grounding DINO loved;
        Gemini is less prone to them but it still happens occasionally."""
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

    def save_overlay(self, bgr, detections):
        """Draw boxes + labels with PIL (no cv2 dependency) and write a PNG.
        Returns the saved path."""
        rgb = bgr[:, :, ::-1]                       # BGR (cv_bridge) -> RGB
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
            draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4],
                           fill=(0, 255, 0))
            draw.text((x1 + 3, ty + 2), tag, fill=(0, 0, 0), font=font)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(self.annotated_dir, f'det_{stamp}.png')
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