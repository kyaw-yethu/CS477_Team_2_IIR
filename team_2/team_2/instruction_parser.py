#!/usr/bin/env python3
"""
instruction_parser.py

STANDBY node. Subscribes to /task_commands (std_msgs/String), parses the
free-form instruction into a list of (object, destination) pairs using a local
Qwen 2.5 GGUF model (llama-cpp-python), and republishes the structured result
as JSON on /parsed_tasks for the perception node to consume.

Example in:  "Move the banana and the meat can to the left storage.
              Move the coke can on the shelf."
Example out: {"tasks": [{"object": "banana",   "destination": "left_storage"},
                        {"object": "meat can", "destination": "left_storage"},
                        {"object": "coke can", "destination": "shelf"}]}
"""
import os
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from std_msgs.msg import String

from llama_cpp import Llama


SYSTEM_PROMPT = """You are a task parser for a robot pick-and-place system.
You receive one natural-language instruction and must extract EVERY
(object, destination) pair it contains.

Valid destinations are EXACTLY one of: "left_storage", "right_storage", "shelf".
Map phrasing as follows:
- "left storage"  / "storage A"  -> "left_storage"
- "right storage" / "storage B"  -> "right_storage"
- "shelf" / "bookshelf"          -> "shelf"

Return ONLY a JSON object of the form:
{"tasks": [{"object": "<name>", "destination": "<left_storage|right_storage|shelf>"}]}

Rules:
- Use lowercase object names, e.g. "coke can", "meat can", "banana", "hammer".
- One entry per (object, destination) occurrence.
- No commentary, no markdown, JSON only."""

VALID_DEST = {"left_storage", "right_storage", "shelf"}


class InstructionParser(Node):
    def __init__(self):
        super().__init__('instruction_parser')
        
        self.declare_parameter(
            'qwen_gguf',
            os.getenv('QWEN_GGUF',
                      '/models/qwen2.5-7b-instruct-q4_k_m.gguf'))
        self.declare_parameter('task_command_topic', '/task_commands')
        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('n_ctx', 4096)
        self.declare_parameter('n_gpu_layers', 0)   # -1 = all layers on GPU
        self.declare_parameter('temperature', 0.0)

        gguf = self.get_parameter('qwen_gguf').value
        n_ctx = int(self.get_parameter('n_ctx').value)
        ngl = int(self.get_parameter('n_gpu_layers').value)
        self.temperature = float(self.get_parameter('temperature').value)

        # Pointing llama.cpp at shard 00001-of-00002 is enough; it auto-loads
        # the remaining shards that sit next to it.
        self.get_logger().info(f'Loading Qwen GGUF: {gguf} (n_gpu_layers={ngl})')
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx,
                         n_gpu_layers=ngl, verbose=False)
        self.get_logger().info('Qwen model loaded.')

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

        self._busy = threading.Lock()
        self.get_logger().info(
            'instruction_parser: STANDBY (waiting on /task_commands)')

    def on_command(self, msg):
        # NOTE: inference runs inline here for simplicity. It blocks this node's
        # executor for a few seconds; fine for a one-shot command. If you later
        # need the node responsive during inference, move parse() onto a worker
        # thread and publish from there.
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

    def parse(self, instruction):
        out = self.llm.create_chat_completion(
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': instruction},
            ],
            response_format={'type': 'json_object'},  # grammar-constrained JSON
            temperature=self.temperature,
        )
        raw = out['choices'][0]['message']['content']
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
