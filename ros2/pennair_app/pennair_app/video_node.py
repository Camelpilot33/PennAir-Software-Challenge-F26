import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


class VideoNode(Node):
    def __init__(self):
        super().__init__("video_node")

        self.publisher = self.create_publisher(Image, "/camera/image", 10)
        self.bridge = CvBridge()
        video_path = Path(get_package_share_directory("pennair_app")) / "input_p5.mp4"
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise RuntimeError("vid not found")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info("video node started")

    def publish_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("video finished")
            self.cap.release()
            self.timer.cancel()
            return
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher.publish(image_msg)

    def destroy_node(self):

        self.cap.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)
    node = VideoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
