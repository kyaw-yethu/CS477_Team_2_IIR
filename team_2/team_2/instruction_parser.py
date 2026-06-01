#!/usr/bin/env python3
"""
instruction_parser.py

STANDBY node. Subscribes to /task_commands (std_msgs/String), parses the
free-form instruction into a list of (object, destination) pairs using the
Gemini API, and republishes the structured result as JSON on /parsed_tasks for
the perception node to consume.

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

from google import genai


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

        self.declare_parameter('model', 'gemini-2.5-flash')
        # If false, skip Gemini and always use the local fallback parser.
        self.declare_parameter('use_gemini', True)
        self.declare_parameter('task_command_topic', '/task_commands')
        self.declare_parameter('parsed_tasks_topic', '/parsed_tasks')
        self.declare_parameter('temperature', 0.0)

        self.model = self.get_parameter('model').value
        self.temperature = float(self.get_parameter('temperature').value)

        self.use_gemini = bool(self.get_parameter('use_gemini').value)
        self._api_keys = self._load_api_keys()
        self._client_index = 0
        # Use a distinct attribute name so we don't shadow rclpy's internal
        # `self._clients` which is used to track ROS clients on the Node.
        if not self.use_gemini:
            self._genai_clients = []
            self.get_logger().warn('use_gemini=false: forcing local fallback parser.')
        elif self._api_keys:
            self._genai_clients = [genai.Client(api_key=key) for key in self._api_keys]
            self.get_logger().info(
                f'Gemini parser ready, model={self.model}, keys={len(self._genai_clients)}')
        else:
            self._genai_clients = []
            self.get_logger().warn(
                'No Gemini API keys provided — using local fallback parser.')

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

    def _load_api_keys(self):
        env_keys = os.getenv('GEMINI_API_KEYS', '')
        print(f"API KEY: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {env_keys}")
        print(f"API KEY: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {env_keys.split(',')}")
        if not env_keys:
            return []
        return env_keys.split(',')

    def _parse_fallback(self, instruction):
        """Heuristic sentence-based parser for offline use.

        Splits the instruction into sentences to avoid cross-sentence
        greedy matches, extracts the object text preceding a destination
        phrase using common prepositions, splits lists (comma/and), and
        returns a deduplicated list of (object,destination) pairs.
        """
        import re

        s = instruction.lower()
        dst_map = {
            'left storage': 'left_storage', 'storage a': 'left_storage',
            'right storage': 'right_storage', 'storage b': 'right_storage',
            'shelf': 'shelf', 'bookshelf': 'shelf'
        }

        tasks = []

        # Process sentence-by-sentence to avoid matching across periods.
        sentences = re.split(r'[\.\?!]+', s)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            matched = False
            # Look for explicit destination mentions in the sentence.
            for dst_phrase, dst_code in dst_map.items():
                # Match patterns like "... <objects> to the left storage"
                m = re.search(
                    rf'(.+?)\b(?:to|onto|in|into|on)\b(?:\s+the\s+)?{re.escape(dst_phrase)}\b',
                    sent)
                if m:
                    objs_text = m.group(1).strip()
                    parts = re.split(r',| and | then | & ', objs_text)
                    for p in parts:
                        p = re.sub(r'^(move|pick up|place|put|grab)\s+(the|a|an)\s+', '', p).strip()
                        if p:
                            tasks.append({'object': p, 'destination': dst_code})
                    matched = True
                    break

                # Also accept '<object> on the shelf' style where object precedes 'on'
                m2 = re.search(
                    rf'\b([a-z0-9\s\-]+?)\b\s+on\s+(?:the\s+)?{re.escape(dst_phrase)}\b',
                    sent)
                if m2:
                    objs_text = m2.group(1).strip()
                    parts = re.split(r',| and | then | & ', objs_text)
                    for p in parts:
                        p = re.sub(r'^(move|pick up|place|put|grab)\s+(the|a|an)\s+', '', p).strip()
                        if p:
                            tasks.append({'object': p, 'destination': dst_code})
                    matched = True
                    break

            # If no explicit destination mention was found in this sentence,
            # skip it (we don't try to infer destinations across sentences).
            if not matched:
                continue

        # Deduplicate and clean object names (remove leading articles, normalize spaces)
        seen = set()
        out = []
        for t in tasks:
            obj = ' '.join(t['object'].split())
            obj = re.sub(r'^(the|a|an)\s+', '', obj)
            key = (obj, t['destination'])
            if key in seen:
                continue
            seen.add(key)
            out.append({'object': obj, 'destination': t['destination']})

        return out

    def _next_client(self):
        client = self._genai_clients[self._client_index]
        self._client_index = (self._client_index + 1) % len(self._genai_clients)
        return client

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
        # If we have Gemini clients, prefer the LLM parse. Otherwise use the
        # offline heuristic parser.
        if not self._genai_clients:
            self.get_logger().info('Parsing instruction with local fallback parser.')
            return self._parse_fallback(instruction)

        last_error = None
        for _ in range(len(self._genai_clients)):
            client = self._next_client()
            try:
                out = client.models.generate_content(
                    model=self.model,
                    contents=[
                        SYSTEM_PROMPT,
                        f'Instruction: {instruction}',
                        'Return only JSON.'
                    ],
                )
                raw = (out.text or '').strip()
                data = json.loads(raw)
                break
            except Exception as e:
                last_error = e
                self.get_logger().warn(
                    f'Gemini parse attempt failed, rotating key: {e}')
        else:
            # Ensure we raise a proper exception type.
            if isinstance(last_error, BaseException):
                raise last_error
            else:
                raise RuntimeError(f'Gemini parse failed: {last_error}')

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