import rospy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, TimeSynchronizer

class TimeSyncListener():
    def __init__(self, model_loader):
        # Initialize the node
        rospy.init_node('time_sync_listener', anonymous=True)

        # Store the model
        self.model_loader = model_loader

        self.intrinsic_matrix = None
        self.camera_info_topic = "/hazard_front/zed_node_front/depth/camera_info"

        # Subscribers for the topics
        self.image_sub = Subscriber('/hazard_front/zed_node_front/left/image_rect_color/compressed', Image)
        self.depth_sub = Subscriber('/hazard_front/zed_node_front/depth/depth_registered', Image)

        # Synchronize the topics using TimeSynchronizer
        self.ts = TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.callback)

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
                self.intrinsic_matrix = message_type.P
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def callback(self, image_msg, depth_msg):
        """
        Callback function that handles synchronized messages.
        """
        if self.intrinsic_matrix is None:
            self.single_listen(self.camera_info_topic, CameraInfo)
        rospy.loginfo("Synchronized messages received")
        rospy.loginfo(f"Image timestamp: {image_msg.header.stamp}")
        rospy.loginfo(f"Depth timestamp: {depth_msg.header.stamp}")

    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
