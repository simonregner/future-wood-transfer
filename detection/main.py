import sys

import rospy

from ros_tools.ros_listener import TimeSyncListener
from detection.models.yolo import YOLOModelLoader

# main start function
if __name__ == "__main__":
    if len(sys.argv) == 1:
        rospy.logwarn("No Max Distance defined - Max Distance will be set to 13 meter")
        max_dist = 13
    else:
        max_dist = int(sys.argv[1])
        rospy.loginfo("Max Distance set to {}".format(max_dist))

    # Load and define model
    model_loader = YOLOModelLoader()
    model_loader.load_model("best.pt")  # Load the YOLOv8 nano model

    ros_listener = TimeSyncListener(model_loader)
    ros_listener.run(max_dist)