#import sys
#sys.path.append('../../')
import rospy

from road_lines_msg.msg import RoadLinesMsg, RoadLine
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import math

import tf
import tf.transformations as tf_transformations


class RoadLinesPublisher:
    def __init__(self, topic_name='/path_array'):
        self.publisher = rospy.Publisher(topic_name, RoadLinesMsg, queue_size=10)
        self.frame_id = None

    def publish_path(self, point_array_left, point_array_right, frame_id):
        self.frame_id = frame_id

        msg = RoadLinesMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id

        for i in range(len(point_array_left)):
            points_left = point_array_left[i]
            points_right = point_array_right[i]
            # Initialize the Path message
            def create_path(points):
                path = Path()
                path.header.stamp = rospy.Time.now()
                path.header.frame_id = self.frame_id

                if len(points) == 0:
                    return path

                # Compute yaw angles and build poses
                for i in range(2, len(points) - 1):
                    current_point = points[i]
                    next_point = points[i + 1]
                    dx, dy = next_point[0] - current_point[0], next_point[1] - current_point[1]
                    yaw = math.atan2(dy, dx)
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

                    # Create PoseStamped message
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = rospy.Time.now()
                    pose_stamped.header.frame_id = self.frame_id
                    pose_stamped.pose.position.x = current_point[0]
                    pose_stamped.pose.position.y = current_point[1]
                    pose_stamped.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
                    pose_stamped.pose.orientation.x = quaternion[0]
                    pose_stamped.pose.orientation.y = quaternion[1]
                    pose_stamped.pose.orientation.z = quaternion[2]
                    pose_stamped.pose.orientation.w = quaternion[3]

                    path.poses.append(pose_stamped)

                return path

            road_line = RoadLine()
            road_line.left_path = create_path(points_left)
            road_line.right_path = create_path(points_right)

            msg.paths.append(road_line)

        self.publisher.publish(msg)
