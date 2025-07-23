import argparse
import rclpy

from ros_tools.ros_listener import TimeSyncListener  # This must inherit from Node!
from detection.models.yolo.yolo import YOLOModelLoader
#from detection.models.mask2former.mask2former import Mask2FormerModelLoader

import yaml


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args, unknown = parser.parse_known_args()

    # Load from config if specified
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    rclpy.init()

    if  hasattr(args, 'model_path'):
        model_path = args.model_path
    else:
        if args.model_type == "yolo":
            model_path = "models/yolo/best.pt"
        elif args.model_type == "mask2former":
            model_path = "models/mask2former/model_final.pth"

    # Load your model
    if args.model_type == "yolo":
        model_loader = YOLOModelLoader()
        model_loader.load_model(model_path)
    #elif args.model_type == "mask2former":
    #    model_loader = Mask2FormerModelLoader()
    #    model_loader.load_model(
    #        "models/mask2former/configs/maskformer2_R50_bs16_50ep.yaml",
    #        model_path
    #    )

    # Create and spin your main ROS2 node
    ros_listener = TimeSyncListener(model_loader, args=args)

    try:
        rclpy.spin(ros_listener)
    except KeyboardInterrupt:
        pass
    finally:
        ros_listener.destroy_node()
        rclpy.shutdown()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()