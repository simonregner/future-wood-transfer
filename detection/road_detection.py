from ultralytics import YOLO
import numpy as np

from ros_tools.ros_listener import TimeSyncListener

debug = False

class YOLOModelLoader:
    """
    A global class to manage the loading and inference of a YOLOv11 model.
    """
    model = None  # Class variable to store the YOLO model instance

    @classmethod
    def load_model(cls, model_path="best.pt"):
        """
        Load the YOLOv11 model if not already loaded.

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
    def predict(cls, image, conf=0.80):
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
        if type(image) is not str:
            image = np.array(image, dtype=np.uint8)
        results = cls.model.predict(source=image, conf=conf)
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
    model_loader.load_model("best.pt")  # Load the YOLOv8 nano model

    if not debug:
        ros_listener = TimeSyncListener(model_loader)
        ros_listener.run()


    if debug:
        from pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask
        from pointcloud.pointcloud_edge_detection import edge_detection

        from path.point_function import fit_line_3d

        import open3d as o3d

        import cv2

        image = cv2.imread("../../ROSBAG_images/ROSBAG_01/images/rgb_1719476048548561819.png")

        results = model_loader.predict(image)  # Replace with your image path

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


        point_cloud = depth_to_pointcloud_from_mask(depth_image='../../ROSBAG_images/ROSBAG_01/depth/depth_1719476048548561819.png', intrinsic_matrix=intrinsic_matrix, mask=mask)
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

        #create_lines_in_image(image=image, lines_left=point_cloud_line_l, lines_right=point_cloud_line_r)


        print("Visualizing the point cloud...")
        #o3d.visualization.draw_geometries([point_cloud, point_cloud_line_l, point_cloud_line_r])

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the geometry to the visualizer
        vis.add_geometry(point_cloud)
        vis.add_geometry(point_cloud_line_l)
        vis.add_geometry(point_cloud_line_r)

        # Get the view control
        view_control = vis.get_view_control()

        # Set the look-at point to (0, 0, 0) and adjust camera parameters
        view_control.set_lookat([0, 0, 0])  # Focus on (0, 0, 0)
        view_control.set_front([0, 0, -1])  # Set the camera front direction
        view_control.set_up([0, -1, 0])  # Set the camera up direction
        view_control.set_zoom(0.5)  # Adjust zoom if needed

        # Run the visualizer
        vis.run()


