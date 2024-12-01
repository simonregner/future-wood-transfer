import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

import numpy as np

from detection.pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask
from detection.pointcloud.pointcloud_edge_detection import edge_detection

from detection.path.point_function import fit_line_3d

from detection.ros_lis.ros_pose import PathPublisher

class TimeSyncListener():
    def __init__(self, model_loader):
        # Initialize the node
        rospy.init_node('time_sync_listener', anonymous=True)

        # Store the model
        self.model_loader = model_loader

        self.intrinsic_matrix = None
        self.camera_info_topic = "/hazard_front/zed_node_front/depth/camera_info"

        self.bridge = CvBridge()

        # Subscribers for the topics
        self.image_sub = Subscriber('/hazard_front/zed_node_front/left/image_rect_color/compressed', CompressedImage)
        self.depth_sub = Subscriber('/hazard_front/zed_node_front/depth/depth_registered', Image)

        # Synchronize the topics using TimeSynchronizer
        self.ts = TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.callback)

        self.left_path_publisher = PathPublisher(node_name='path_left_node', topic_name='/path/left_path', frame_id='map')
        self.right_path_publisher = PathPublisher(node_name='path_right_node', topic_name='/path/right_path', frame_id='map')

        rospy.loginfo("Time sync listener initialized and running")

    def single_listen(self, topic_name, message_type):
        """
        Listens to a topic once and processes the received message.

        :param topic_name: Name of the topic to listen to.
        :param message_type: Type of the ROS message expected.
        """
        rospy.loginfo(f"Listening to topic {topic_name} once.")

        try:
            # Wait for a single message on the topic
            msg = rospy.wait_for_message(topic_name, message_type, timeout=5)
            rospy.loginfo(f"Received a message from {topic_name}")

            # Example processing of the message (convert to OpenCV if it's an Image)
            if message_type == CameraInfo:
                self.intrinsic_matrix = np.array(msg.K).reshape(3, 3)

                print(self.intrinsic_matrix)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def callback(self, image_msg, depth_msg):
        """
        Callback function that handles synchronized messages.
        """
        if self.intrinsic_matrix is None:
            self.single_listen(self.camera_info_topic, CameraInfo)

        # Check if the timestamp is the same
        if image_msg.header.stamp.secs != depth_msg.header.stamp.secs:
            rospy.loginfo("NO synchronized messages received")
            rospy.loginfo(f"Image timestamp: {image_msg.header.stamp}")
            rospy.loginfo(f"Depth timestamp: {depth_msg.header.stamp}")
            return

        rgb_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        depth_image = np.array(depth_image, dtype=np.float32)

        # Replace NaN values with 0 (or another appropriate value)
        depth_array = np.nan_to_num(depth_image, nan=0.0)

        # Clip negative values to 0
        depth_array[depth_array < 0] = 0

        results = self.model_loader.predict(rgb_image)
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

        point_cloud = depth_to_pointcloud_from_mask(
            depth_image=depth_image,
            intrinsic_matrix=self.intrinsic_matrix, mask=mask)
        point_cloud, left_points, right_points = edge_detection(point_cloud=point_cloud)

        if len(left_points) == 0 or len(right_points) == 0:
            return

        x_fine_l, y_fine_l, z_fine_l = fit_line_3d(points=left_points, degree=6)
        x_fine_r, y_fine_r, z_fine_r = fit_line_3d(points=right_points, degree=6)

        points_fine_l = np.vstack((x_fine_l, y_fine_l, z_fine_l)).T
        points_fine_r = np.vstack((x_fine_r, y_fine_r, z_fine_r)).T

        self.left_path_publisher.publish_path(points_fine_l)
        self.right_path_publisher.publish_path(points_fine_r)


    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
