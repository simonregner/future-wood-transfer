import argparse
import yaml
import rclpy

from ros.listener import ObjectDetectionListener
from models.yolo.yolo import YOLODetectionLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    args, _ = parser.parse_known_args()

    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)

    rclpy.init()

    model_path = getattr(args, 'model_path', 'models/yolo/YOLO26s-seg.pt')

    imgsz = getattr(args, 'imgsz', 640)
    half  = getattr(args, 'half', False)

    model_loader = YOLODetectionLoader()
    model_loader.load_model(model_path, imgsz=imgsz, half=half)

    node = ObjectDetectionListener(model_loader, args=args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
