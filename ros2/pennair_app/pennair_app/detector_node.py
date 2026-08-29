import json

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

from .detector import process_image


class DetectorNode(Node):

    def __init__(self):
        super().__init__("detector_node")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, "/camera/image", self.image_callback, 10
        )

        self.publisher = self.create_publisher(String, "/detections", 10)

        self.get_logger().info("Detector node started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        #run algo
        detections = process_image(frame)

        #message
        detection_msg = String()
        detection_msg.data = json.dumps(detections)

        self.publisher.publish(detection_msg)


def main(args=None):

    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
