import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose

from rclpy.duration import Duration

from road_lines_msg.msg import RoadLinesMsg, RoadLine

from nav_msgs.msg import Path

import numpy as np
from sklearn.decomposition import PCA

import tf_transformations  # Note: install 'tf_transformations' Python package!
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math

class RoadLinesMergingListener(Node):
    def __init__(self):
        super().__init__('time_sync_listener')

        self.use_sim_time = True

        self.left_road_line = Path()
        self.right_road_line = Path()

        self.robot_position = [0.0, 0.0, 0.0]  # Initialize robot position

        # TF2 Buffer + Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscrioter = self.create_subscription(
            RoadLinesMsg,
            '/road/road_lines_topic',
            self.road_lines_callback,
            10
        )

        self.robot_odom_subscriber = self.create_subscription(
            Odometry,
            '/mercator_odom',
            self.robot_odom_callback,
            10
        )

        self.road_line_publisher = self.create_publisher(RoadLinesMsg, '/road/road_lines_topic_odom', 10)

        self.left_path_publisher = self.create_publisher(Path, '/path/left_path', 10)
        self.right_path_publisher = self.create_publisher(Path, '/path/right_path',10)


    def tranform_pose(self, path, target_frame='odom', source_frame='base_link'):
        """
        Transform a pose from source_frame to target_frame using the TF2 buffer.
        """

        new_path = Path()
        new_path.header.stamp = path.header.stamp
        new_path.header.frame_id = target_frame 
        new_path.poses = path.poses.copy()  # Create a copy of the poses to transform
        

        for i, pose_stamped in enumerate(path.poses):
            try:
                # Transform the pose from source_frame to target_fram - Source frame from camera - Target frame is the odom frame
                transform = self.tf_buffer.lookup_transform(
                    target_frame=target_frame,  # Target frame
                    source_frame=source_frame,  # Source frame
                    time=pose_stamped.header.stamp,  # Use the timestamp from the pose header
                    #time=self.get_clock().now(),  # Use the current time
 # Use the timestamp from the path header
                    timeout=Duration(seconds=0.5)
                )

                transformed_pose = do_transform_pose(pose_stamped.pose, transform)

                new_path.poses[i].pose = transformed_pose
        
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"Transform failed: {str(e)}")
                return None
            
        return new_path
    
    def road_lines_callback(self, msg):

        self.get_logger().info(f"Received RoadLinesMsg with {len(msg.paths)} paths")

        msg_pub = RoadLinesMsg()
        msg_pub.header.stamp = msg.header.stamp 
        msg_pub.header.frame_id = msg.header.frame_id

        for path_index, path in enumerate(msg.paths):
            #print(f"Processing path {path_index} with {len(path.left_path.poses)} left poses and {len(path.right_path.poses)} right poses")

            left_transformed_pose = self.tranform_pose(path.left_path, target_frame='odom', source_frame=msg.header.frame_id)
            right_transformed_pose = self.tranform_pose(path.right_path, target_frame='odom', source_frame=msg.header.frame_id)

            if left_transformed_pose is None or right_transformed_pose is None:
                return
            
            left_new_pose = self.merge_new_old_paths(left_transformed_pose, self.left_road_line)
            right_new_pose = self.merge_new_old_paths(right_transformed_pose, self.right_road_line)

            road_line = RoadLine()
            road_line.left_path = left_new_pose
            road_line.right_path = right_new_pose

            self.left_road_line = left_new_pose
            self.right_road_line = right_new_pose

            road_line.left_path.header.frame_id = 'odom'
            road_line.right_path.header.frame_id = 'odom'

            road_line.left_path.header.stamp = msg.header.stamp
            road_line.right_path.header.stamp = msg.header.stamp

            msg_pub.paths.append(road_line)

        self.road_line_publisher.publish(msg)

        self.left_path_publisher.publish(msg_pub.paths[0].left_path)
        self.right_path_publisher.publish(msg_pub.paths[0].right_path)


    def merge_new_old_paths(self, new_path, old_path):
        """        Merges two paths by appending the new path to the old path.
        """
        poses = old_path.poses
        points_old = np.array([[pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z] for pose_stamped in poses])

        poses = new_path.poses

        print(poses[0])

        points_new = np.array([[pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z] for pose_stamped in poses])


        print(f"Old path points: {points_old.shape}, New path points: {points_new.shape}")
  

        if points_old.shape[0] == 0:
            combined_points = points_new
        elif points_new.shape[0] == 0:
            combined_points = points_old
        else:
            combined_points = np.vstack((points_new, points_old))

        print(f"Combined points shape: {combined_points.shape}")

        x_fit, y_fit, z_fit = calculate_fitted_line(combined_points, degree=2, t_fit=100)

        points = np.array([x_fit, y_fit, z_fit]).T

        print(f"Fitted points shape: {points.shape}")

        # Compute Euclidean distances from the first point
        distances = np.linalg.norm(points - self.robot_position, axis=1)

        print(f"Distances from robot position: {distances}")

        # Keep only points within 20 meters
        points = points[distances <= 100.0]

        print(f"Filtered points shape: {points.shape}")

        poses = []
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]

            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            dz = next_point[2] - current_point[2]

            yaw = math.atan2(dy, dx)
            distance_xy = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.atan2(-dz, distance_xy)  # Negative dz for ROS

            # Convert to quaternion: (roll, pitch, yaw)
            quaternion = tf_transformations.quaternion_from_euler(0, pitch, yaw)

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = new_path.header.frame_id
            pose_stamped.pose.position.x = current_point[0]
            pose_stamped.pose.position.y = current_point[1]
            pose_stamped.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]

            poses.append(pose_stamped)

        new_path.poses = poses

        return new_path


    def robot_odom_callback(self, msg):
        """
        Callback function to handle robot odometry messages.
        This function updates the robot's position based on the received odometry data.
        """

        #print("Robot: ", msg)
        self.robot_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        #self.get_logger().info(f"Updated robot position: {self.robot_position}")
   


def calculate_fitted_line(points, degree, t_fit):
    # PCA to sort points
    t = PCA(n_components=1).fit_transform(points).flatten()
    sorted_idx = np.argsort(t)
    points_sorted = points[sorted_idx]
    t_sorted = t[sorted_idx]

    # Fit polynomial for each coordinate using t_sorted as the parameter:
    # (you might want to adjust degrees as needed)
    poly_x = np.polyfit(t_sorted, points_sorted[:, 0], degree)
    poly_y = np.polyfit(t_sorted, points_sorted[:, 1], 2)
    poly_z = np.polyfit(t_sorted, points_sorted[:, 2], degree)

    # Generate the main fitted t-values for your current range:
    t_fit_main = np.linspace(t_sorted[0], t_sorted[-1], t_fit)
    x_fit_main = np.polyval(poly_x, t_fit_main)
    y_fit_main = np.polyval(poly_y, t_fit_main)
    # y_fit_main = np.zeros(t_fit_len)
    z_fit_main = np.polyval(poly_z, t_fit_main)

    # Estimate a spacing for the t-values.
    # One simple approach is to use the difference between the first two sorted t-values.
    # (You may also compute the median of t differences if that's more robust.)
    dt = t_sorted[1] - t_sorted[0]

    n_extension = 0

    # Generate n_extension extra t-values that extend *before* the beginning of your data.
    # For example, if you have 5 extra points, you can create them from t_sorted[0] - 5*dt up to t_sorted[0]
    t_fit_ext = np.linspace(t_sorted[0] - n_extension * dt, t_sorted[0], n_extension, endpoint=False)

    # Evaluate the fitted polynomial on the extension t-values:
    x_fit_ext = np.polyval(poly_x, t_fit_ext)
    y_fit_ext = np.polyval(poly_y, t_fit_ext)
    z_fit_ext = np.polyval(poly_z, t_fit_ext)

    # Option 2: If you want to combine the extension and the fitted points:
    x_fit = np.concatenate([x_fit_ext, x_fit_main])
    y_fit = np.concatenate([y_fit_ext, y_fit_main])
    z_fit = np.concatenate([z_fit_ext, z_fit_main])

    return x_fit, y_fit, z_fit


if __name__ == "__main__":
    rclpy.init()
    listener = RoadLinesMergingListener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()