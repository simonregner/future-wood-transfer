from ultralytics import YOLO
import numpy as np

from ros_tools.ros_listener import TimeSyncListener

import matplotlib.pyplot as plt

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
    def predict(cls, image, conf=0.50):
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
        results = cls.model.predict(source=image, conf=conf, retina_masks=True)#, agnostic_nms=True, retina_masks=True)
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


# main start function
if __name__ == "__main__":

    # Load and define model
    model_loader = YOLOModelLoader()
    model_loader.load_model("best.pt")  # Load the YOLOv8 nano model

    if not debug:
        ros_listener = TimeSyncListener(model_loader)
        ros_listener.run()


    # Only for debugging (probably not working code)
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

        print("BOXES")
        print(results[0].boxes.xyxy[0].cpu().numpy())

        box_value = results[0].boxes.xyxy[0].cpu().numpy()


        def find_last_mask_pixel(mask, col, bottom, top):
            """
            mask: 2D numpy array with 0 and 255 values.
            col: the column index at which to search (either left or right boundary of the bounding box).
            bottom: row index of the bottom boundary of the bounding box.
            top: row index of the top boundary (usually 0 or the top of the bounding box).
            Returns the row index of the last mask pixel (255) when scanning upward.
            """
            last_mask_row = None
            for row in range(bottom, top - 1, -1):  # move upward
                if mask[row, col] == 255:
                    last_mask_row = row
                else:
                    # As soon as you hit a non-mask pixel, stop scanning.
                    break
            return last_mask_row


        left_mask = find_last_mask_pixel(mask, col=int(box_value[0]) + 1, bottom=int(box_value[3]) - 1, top=int(box_value[1]) + 1)
        right_mask = find_last_mask_pixel(mask, col=int(box_value[2]) - 1, bottom=int(box_value[3]) - 1, top=int(box_value[1]) + 1)
        print("Mask END: ", left_mask, right_mask)

        plt.imshow(mask, cmap="gray")
        plt.show()


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


