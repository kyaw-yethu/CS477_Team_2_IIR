import sys
import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose
from geometry_msgs.msg import Pose

class ObjectDetectClient(Node):

    def __init__(self):
        super().__init__('string_pose_client')
        # Service client
        self.cli = self.create_client(StringPose, 'detect_objects_with_prompt')
        
        # Waiting a service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service is not available. Waiting...')
        
        self.req = StringPose.Request()

    def send_request(self, text):
        self.req.data = text
        
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    
    client = ObjectDetectClient()
    
    response = client.send_request("Detect a meat_can and return [ymin, xmin, ymax, xmax, label]")
    
    if response is not None:
        client.get_logger().info('Service call success!')
    else:
        client.get_logger().error('Service call failed!!')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
