from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    video_node = Node(
        package="pennair_app",
        executable="video_node",
        name="video_node",
        output="screen",
    )

    detector_node = Node(
        package="pennair_app",
        executable="detector_node",
        name="detector_node",
        output="screen",
    )

    return LaunchDescription([
        video_node,
        detector_node,
    ])