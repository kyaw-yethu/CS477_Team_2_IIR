import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion, Pose
from riro_srvs.srv import StringPose
from cv_bridge import CvBridge
from google import genai
from PIL import Image as PILImage
import cv2
import re
from rclpy.qos import QoSProfile, ReliabilityPolicy

class GeminiPromptServiceNode(Node):
    def __init__(self):
        super().__init__('gemini_prompt_service_node')
        
        # 1. API Setup
        self.declare_parameter('api_key', "AIzaSyDU3CEUJZNqluNDmM8XT2b_ovv8AaJ-uo8")
        api_key = self.get_parameter('api_key').get_parameter_value().string_value

        self.client = genai.Client(api_key=api_key)
        self.model  = 'gemini-2.5-flash'

        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(f"Available Model: {m.name}")
        
        # 2. QoS Setup
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        # 3. Setup the subscriber for the image topic
        self.subscription = self.create_subscription(
            Image,
            '/wrist_camera/wrist_camera/color/image_raw',
            self.image_callback,
            qos_profile)
        
        # 4. Setup the prompt service
        self.srv = self.create_service(StringPose, 'detect_objects_with_prompt', self.detect_callback)

        # Timer (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.bridge = CvBridge()
        self.latest_cv_img = None
        self.cv_img = None
        self.get_logger().info('Gemini Prompt Service Node is Ready!')

    def image_callback(self, msg):
        """Image subscriber callback"""
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def timer_callback(self):
        """Timer callback"""
        if self.cv_img is not None:
            cv2.imshow("Detection Result", self.cv_img)
            cv2.waitKey(1)
        
    def detect_callback(self, request, response):
        """
        VLM service callback that returns the object location in the image 
        given the user prompt 
        """
        user_prompt = request.data 
        
        if not user_prompt:
            user_prompt = "Detect objects and return [ymin, xmin, ymax, xmax, label]"
            
        if self.latest_cv_img is None:
            response.success = False
            response.message = "No camera image received."
            return response

        try:
            self.get_logger().info(f"Received Prompt: {user_prompt}")
            
            # image conversion
            rgb_img = cv2.cvtColor(self.latest_cv_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_img)

            # Gemini call
            format_instruction = (
                "\n\nOutput format: [ymin, xmin, ymax, xmax, 'label'] "
                "using normalized coordinates (0-1000). "
                "Only return the list, no other text."
            )

            vlm_response = self.client.models.generate_content(
                model=self.model,
                contents=[pil_img, user_prompt+ format_instruction])
            result_text = vlm_response.text
            self.get_logger().info(f"Received Result: {result_text}")
            
            # visualization
            vis_img = self.latest_cv_img.copy()
            h, w, _ = vis_img.shape
            
            # extract the coordinate
            pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*["\']?([\w\s]+)["\']?\]'
            matches = re.findall(pattern, result_text)
            self.get_logger().info(f"Found {len(matches)} objects.")

            ymin, xmin, ymax, xmax = 0,0,0,0
            match = matches[0]

            # get integer values and a label
            ymin, xmin, ymax, xmax, label = int(match[0]), int(match[1]), int(match[2]), int(match[3]), match[4].strip()
                    
            # get pixel coordiantes
            left, top = int(xmin * w / 1000), int(ymin * h / 1000)
            right, bottom = int(xmax * w / 1000), int(ymax * h / 1000)
            
            # get the center coordinate
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            # draw a bounding Box
            cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # draw a center (red)x
            cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
            # overlay the label and the coordinate
            display_text = f"{label} ({center_x}, {center_y})"
            cv2.putText(vis_img, display_text, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.cv_img = vis_img.copy()

            self.get_logger().info(f"Target Center: x={center_x}, y={center_y}")
        
            p = Pose()
            p.position.x = float(center_x)
            p.position.y = float(center_y)
            response.pose = p

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            response.pose = Pose()

        return response

def main():
    rclpy.init()
    node = GeminiPromptServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    
