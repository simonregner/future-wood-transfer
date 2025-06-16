import sys
import rclpy
from rclpy.node import Node

from ros_tools.ros_listener import TimeSyncListener  # This must inherit from Node!
from detection.models.yolo.yolo import YOLOModelLoader

def main(args=None):
    rclpy.init(args=args)

    # Get max distance from command line or use default
    if len(sys.argv) == 1:
        max_dist = 13
        print("[WARN] No Max Distance defined - Max Distance will be set to 13 meter")
    else:
        max_dist = float(sys.argv[1])
        print(f"[INFO] Max Distance set to {max_dist}")

    # Load your model
    model_loader = YOLOModelLoader()
    model_loader.load_model("best.pt")

    # Create and spin your main ROS2 node
    ros_listener = TimeSyncListener(model_loader)
    ros_listener.max_depth = max_dist   # Set your max distance

    try:
        rclpy.spin(ros_listener)
    except KeyboardInterrupt:
        pass
    finally:
        ros_listener.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()