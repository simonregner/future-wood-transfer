import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
import tf.transformations as tf_transformations
import numpy as np

def compute_orientation(point_a, point_b):
    direction = np.array(point_b) - np.array(point_a)
    yaw = np.arctan2(direction[1], direction[0])
    quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])


class PathPublisher:
    def __init__(self, points):
        rospy.init_node('path_publisher', anonymous=True)
        self.publisher_ = rospy.Publisher('path', Path, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_path)
        self.points = points

    def publish_path(self, event):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = rospy.Time.now()

        for i in range(len(self.points) - 1):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = self.points[i][0]
            pose.pose.position.y = self.points[i][1]
            pose.pose.position.z = self.points[i][2]

            orientation = compute_orientation(self.points[i], self.points[i + 1])
            pose.pose.orientation = orientation

            path_msg.poses.append(pose)

        # Add the last point without orientation
        last_pose = PoseStamped()
        last_pose.header.frame_id = 'map'
        last_pose.header.stamp = rospy.Time.now()
        last_pose.pose.position.x = self.points[-1][0]
        last_pose.pose.position.y = self.points[-1][1]
        last_pose.pose.position.z = self.points[-1][2]
        path_msg.poses.append(last_pose)

        self.publisher_.publish(path_msg)
        rospy.loginfo('Path published with %d poses' % len(path_msg.poses))

