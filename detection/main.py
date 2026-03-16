import argparse
import rclpy

from ros.listener import TimeSyncListener
from models.yolo.yolo import YOLOModelLoader
from models.mask2former.mask2former import Mask2FormerModelLoader

import yaml


def main():
    # --- Argument Parsing ---
    # Accepts an optional --config flag pointing to a YAML config file.
    # All other unknown args (e.g. ROS2 remappings) are silently ignored.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args, unknown = parser.parse_known_args()

    # If a config file is provided, load it and merge its keys into args,
    # so config values are accessible the same way as CLI arguments.
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    # --- ROS2 Initialization ---
    # Must be called before any ROS2 node is created.
    rclpy.init()

    # --- Model Path Resolution ---
    # Use the explicit model_path from config/CLI if given,
    # otherwise fall back to the default path for the chosen model type.
    if hasattr(args, 'model_path'):
        model_path = args.model_path
    else:
        if args.model_type == "yolo":
            model_path = "models/yolo/best.pt"
        elif args.model_type == "mask2former":
            model_path = "models/mask2former/model_final.pth"

    # --- Model Loading ---
    # Instantiate the correct model loader based on model_type and load weights.
    # YOLO only needs the weights file; Mask2Former also requires a config yaml.
    if args.model_type == "yolo":
        model_loader = YOLOModelLoader()
        model_loader.load_model(model_path)
    elif args.model_type == "mask2former":
        model_loader = Mask2FormerModelLoader()
        model_loader.load_model(
            "models/mask2former/configs/maskformer2_R50_bs16_50ep.yaml",
            model_path
        )

    # --- ROS2 Node Startup ---
    # TimeSyncListener subscribes to camera topics, runs inference with the
    # loaded model, and publishes detection results. rclpy.spin() keeps it
    # alive, processing callbacks until the user presses Ctrl+C.
    ros_listener = TimeSyncListener(model_loader, args=args)

    try:
        rclpy.spin(ros_listener)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up the node and shut down the ROS2 context on exit.
        ros_listener.destroy_node()
        rclpy.shutdown()


def load_config(path):
    # Read and parse a YAML config file, returning it as a plain dict.
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()