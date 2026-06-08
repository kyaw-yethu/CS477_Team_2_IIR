#!/usr/bin/env python3
"""
instruction_parser.py

STANDBY node. Subscribes to /task_commands (std_msgs/String), parses the
free-form instruction into a list of (object, destination) pairs using the
Gemini API (google-genai), and republishes the structured result as JSON on
/parsed_tasks for the perception node to consume.

Example in:  "Move the banana and the meat can to the left storage.
              Move the coke can on the shelf."
Example out: {"tasks": [{"object": "banana",   "destination": "left_storage"},
                        {"object": "meat can", "destination": "left_storage"},
                        {"object": "coke can", "destination": "shelf"}]}

Set your API key before launching. Gemini is used if GEMINI_API_KEY is present;
otherwise it falls back to OpenAI via OPENAI_API_KEY:
    export GEMINI_API_KEY="your_api_key"      # preferred backend
    export OPENAI_API_KEY="your_api_key"      # fallback backend
or pass it as the 'api_key' / 'openai_api_key' ROS parameter.
"""
import os
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from std_msgs.msg import String
from sensor_msgs.msg import Image

import numpy as np
from PIL import Image as PILImage
from cv_bridge import CvBridge

# Either backend may be absent; import lazily so the node runs on whichever is
# installed + keyed.
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """
You are a task parser for a robot pick-and-place system. You receive one natural-language instruction and one image of objects to be picked. 
Note that multiple instances of the same object category may be present in the workspace. So, carefully analyzing the instruction and the image, identify which 
specific objects to pick and where to place them.

Valid destinations are EXACTLY one of: "left_storage", "right_storage", "shelf".
Map phrasing as follows:
- "left storage"  / "storage A"  -> "left_storage"
- "right storage" / "storage B"  -> "right_storage"
- "shelf" / "bookshelf"          -> "shelf"

Return ONLY a JSON object of the form:
{"tasks": [{"object": "<name>", "destination": "<left_storage|right_storage|shelf>"}]}

If there are multiple instances of the same object according to the instruction and the image, {"object": "<name>", "destination": "<left_storage|right_storage|shelf>"} must be repeated for each instance.

Rules:
- Use lowercase object names, e.g. "coke can", "meat can", "banana", "hammer".
- One entry per (object, destination) occurrence.
- No commentary, no markdown, JSON only.
- If the noun is plural, assume all instances of that object category are meant. 
- If the noun is singular, assume only one instance is meant.
- If the number of a particular object category is specified in the instruction, assume that exact number of instances are meant.
- No ordering among objects should be assumed unless explicitly stated in the instruction.
"""

VALID_DEST = {"left_storage", "right_storage", "shelf"}


class InstructionParser(Node):
    def __init__(self):
        super().__init__('instruction_parser')

        # ---- Parameters --------------------------------------------------
        # Declare up front (the old code read 'api_key' before declaring it,
        # which raised whenever GEMINI_API_KEY was unset). Backend chosen below.
        self.declare_parameter('api_key', '')           # Gemini key (env wins)
        self.declare_parameter('openai_api_key', '')    # OpenAI key (env wins)
        self.declare_parameter('model', 'gemini-3-flash-preview')
        self.declare_parameter('openai_model', 'gpt-4o-mini')
        self.declare_parameter('task_command_topic', '/task_commands')
        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('wrist_image_topic',
                               '/wrist_camera/wrist_camera/color/image_raw')
        self.declare_parameter('temperature', 0.0)

        self.temperature = float(self.get_parameter('temperature').value)

        # ---- Backend selection -------------------------------------------
        # Gemini wins if its key is present; otherwise fall back to OpenAI.
        # Env vars take precedence over the corresponding ROS params.
        gemini_key = os.getenv('GEMINI_API_KEY') \
            or self.get_parameter('api_key').get_parameter_value().string_value
        openai_key = os.getenv('OPENAI_API_KEY') \
            or self.get_parameter('openai_api_key').get_parameter_value().string_value

        if gemini_key:
            if genai is None:
                raise ImportError(
                    "GEMINI_API_KEY is set but the 'google-genai' package is "
                    'not installed in this image.')
            self.backend = 'gemini'
            self.model = self.get_parameter('model').value
            self.client = genai.Client(api_key=gemini_key)
        elif openai_key:
            if OpenAI is None:
                raise ImportError(
                    "OPENAI_API_KEY is set but the 'openai' package is not "
                    'installed in this image.')
            self.backend = 'openai'
            self.model = self.get_parameter('openai_model').value
            self.client = OpenAI(api_key=openai_key)
        else:
            raise ValueError(
                'No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY as an '
                "environment variable, or provide the 'api_key' / "
                "'openai_api_key' ROS parameter.")

        self.get_logger().info(f'Parser backend: {self.backend}, model: {self.model}')

        # ---- QoS ---------------------------------------------------------
        # Must match the publisher in example3_task_command_pub (RELIABLE).
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Latched so a perception node that joins late still gets the last parse.
        out_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub = self.create_publisher(
            String, self.get_parameter('parsed_tasks_topic').value, out_qos)
        self.create_subscription(
            String, self.get_parameter('task_command_topic').value,
            self.on_command, cmd_qos)
        
        self.bridge = CvBridge()
        self.latest_image = None
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image, self.get_parameter('wrist_image_topic').value,
            self.on_image, img_qos)

        self._busy = threading.Lock()
        self.get_logger().info(
            'instruction_parser: STANDBY (waiting on /task_commands)')

    def on_command(self, msg):
        # NOTE: the Gemini call runs inline here for simplicity. It blocks this
        # node's executor for the duration of the request; fine for a one-shot
        # command. If you later need the node responsive during inference, move
        # parse() onto a worker thread and publish from there.
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already parsing a command; ignoring new one.')
            return
        try:
            self.get_logger().info(f'Instruction received: {msg.data}')
            tasks = self.parse(msg.data)
            out = String()
            out.data = json.dumps({'tasks': tasks})
            self.pub.publish(out)
            self.get_logger().info(f'Parsed {len(tasks)} task(s): {out.data}')
        except Exception as e:
            self.get_logger().error(f'Parse failed: {e}')
        finally:
            self._busy.release()

    def on_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    def _latest_pil(self):
        """Most recent wrist frame as a PIL RGB image, or None if none received."""
        frame = self.latest_image
        if frame is None:
            return None
        rgb = frame[:, :, ::-1]                      # BGR (cv_bridge) -> RGB
        return PILImage.fromarray(np.ascontiguousarray(rgb))

    def parse(self, instruction):
        # Both backends are prompted to return the same JSON object; only the
        # raw-text generation differs. Validation below is shared.
        if self.backend == 'gemini':
            raw = self._parse_gemini(instruction)
        else:
            raw = self._parse_openai(instruction)
        data = json.loads(raw)

        tasks = []
        for t in data.get('tasks', []):
            obj = str(t.get('object', '')).strip().lower()
            dest = str(t.get('destination', '')).strip().lower()
            if obj and dest in VALID_DEST:
                tasks.append({'object': obj, 'destination': dest})
            else:
                self.get_logger().warn(f'Dropping invalid task entry: {t}')
        return tasks

    def _parse_gemini(self, instruction):
        pil = self._latest_pil()
        contents = [pil, instruction]
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                response_mime_type='application/json',  # force JSON output
            ),
        )
        return response.text

    def _parse_openai(self, instruction):
        # json_object mode requires the word "json" somewhere in the messages;
        # SYSTEM_PROMPT already says "Return ONLY a JSON object", so it's covered.
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': instruction},
            ],
        )
        return resp.choices[0].message.content


def main():
    rclpy.init()
    node = InstructionParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()