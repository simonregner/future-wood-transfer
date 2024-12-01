from ultralytics import YOLO

import numpy as np
import open3d as o3d

from pointcloud.depth_to_pointcloud import depth_to_pointcloud, depth_to_pointcloud_from_mask
from pointcloud.pointcloud_edge_detection import edge_detection

from path.point_function import fit_line_3d

from ros.ros_listener import TimeSyncListener


class YOLOModelLoader:
    """
    A global class to manage the loading and inference of a YOLOv8 model.
    """
    model = None  # Class variable to store the YOLO model instance

    @classmethod
    def load_model(cls, model_path="../yolo_V11/runs/segment/train/weights/best.pt"):
        """
        Load the YOLOv8 model if not already loaded.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        if cls.model is None:
            print(f"Loading YOLOv11 model from {model_path}...")
            cls.model = YOLO(model_path)
            print("Model loaded successfully.")
        else:
            print("Model is already loaded.")

    @classmethod
    def predict(cls, image_path, conf=0.25):
        """
        Perform inference using the loaded YOLO model.

        Args:
            image_path (str): Path to the image to run inference on.
            conf (float): Confidence threshold for predictions.

        Returns:
            results: YOLO predictions.
        """
        if cls.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")

        #print(f"Running inference on {image_path}...")
        results = cls.model.predict(source=image_path, conf=conf, device="0")
        return results

def show_results(results):
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="test_results/result_cross_01.jpg")


# Example usage
if __name__ == "__main__":
    # Load and define model
    model_loader = YOLOModelLoader()
    model_loader.load_model("../yolo_V11/runs/segment/train/weights/best.pt")  # Load the YOLOv8 nano model
    results = model_loader.predict("../../ROSBAG_images/ROSBAG_01/images/rgb_1719476030147187138.png")  # Replace with your image path

    ros_listener = TimeSyncListener(model_loader)

    ros_listener.run()

    # Display results
    show_results(results)  # Show predictions
    print(results)  # Print predictions

    # TODO: Change the code, that i will read the camera matrix from the ROSBAG file
    intrinsic_matrix = np.array([
        [541.736083984375, 0.0, 642.0556640625,],           # fx, 0, cx
        [0.0, 541.736083984375, 347.4380187988281],         # 0, fy, cy
        [0.0, 0.0, 1.0]                                     # 0,  0,  1
    ])

    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

    point_cloud = depth_to_pointcloud_from_mask(depth_image_path='../../ROSBAG_images/ROSBAG_01/depth/depth_1719476030147187138.png', intrinsic_matrix=intrinsic_matrix, mask=mask)
    point_cloud, left_points, right_points = edge_detection(point_cloud=point_cloud)

    x_fine_l, y_fine_l, z_fine_l = fit_line_3d(points=left_points, degree=6)
    x_fine_r, y_fine_r, z_fine_r = fit_line_3d(points=right_points, degree=6)

    print(z_fine_l)


    # Add points from the fitted curve to a point cloud in Open3D
    points_fine_l = np.vstack((x_fine_l, y_fine_l, z_fine_l)).T
    point_cloud_line_l = o3d.geometry.PointCloud()
    point_cloud_line_l.points = o3d.utility.Vector3dVector(points_fine_l)

    # Add points from the fitted curve to a point cloud in Open3D
    points_fine_r = np.vstack((x_fine_r, y_fine_r, z_fine_r)).T
    point_cloud_line_r = o3d.geometry.PointCloud()
    point_cloud_line_r.points = o3d.utility.Vector3dVector(points_fine_r)


    print("Visualizing the point cloud...")
    o3d.visualization.draw_geometries([point_cloud, point_cloud_line_l, point_cloud_line_r])


