debug = False

import numpy as np

from ros_tools.ros_listener import TimeSyncListener

import matplotlib.pyplot as plt

from detection.models.yolo import YOLOModelLoader

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
        import pyransac3d as pyrsc

        image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_02/images/rgb_1737367378279403926.png")
        image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_02/images/rgb_1737367361885361311.png")
        depth_image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_02/depth/depth_1737367361885361311.png", cv2.IMREAD_UNCHANGED)

        image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_03/images/rgb_1738573843100461453.png")
        depth_image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_03/depth/depth_1738573843100461453.png", cv2.IMREAD_UNCHANGED)

        image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_04/images/rgb_1738577274038472281.png")
        depth_image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_04/depth/depth_1738577274038472281.png", cv2.IMREAD_UNCHANGED)

        #image = cv2.imread("../../ROSBAG_images/ROSBAG_UNI_04/images/rgb_1738577274038472281.png")






        results = model_loader.predict(image)  # Replace with your image path

        # Display results
        show_results(results)  # Show predictions
        #print(results)  # Print predictions

        # TODO: Change the code, that i will read the camera matrix from the ROSBAG file
        intrinsic_matrix = np.array([
            [541.736083984375, 0.0, 642.0556640625,],           # fx, 0, cx
            [0.0, 541.736083984375, 347.4380187988281],         # 0, fy, cy
            [0.0, 0.0, 1.0]                                     # 0,  0,  1
        ])

        dist_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        box_value = results[0].boxes.xyxy[0].cpu().numpy()

        # Define the kernel size (10 pixels erosion means a radius of 10 pixels)
        kernel_size = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1))

        # Apply erosion
        test = cv2.erode(mask, kernel, iterations=1)

        kernel_size = 5

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Dilate and erode mask to get edges
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)

        # Subtract eroded from dilated to get outline
        outline = cv2.absdiff(dilated, eroded)

        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        depth_scale = 0.001  # Adjust based on your depth sensor (typically meters)

        # --- Convert depth image to 3D point cloud ---
        h, w = depth_image.shape
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

        # Apply mask to filter depth values
        mask_bool = mask > 0  # Convert mask to boolean

        z = depth_image * depth_scale

        valid_condition = (mask_bool) & (z > 0) & (z < 13)

        z[~valid_condition] = 0  # Set depth outside mask to zero


        x = (xmap - cx) * z / fx
        y = (ymap - cy) * z / fy

        points_3D = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Filter out invalid points
        valid_points = (z > 0).reshape(-1)
        points_3D = points_3D[valid_points]

        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(points_3D, thresh=0.05, minPoints=3, maxIteration=100)

        points_3D = points_3D[best_inliers]

        # --- Bird's-eye view (BEV) coordinate transformation ---
        # Person-view axes: X-right, Y-down, Z-forward
        # Bird's-eye view axes: X-right, Y-forward, Z-up
        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]])

        bev_points = points_3D @ R.T

        # Define bird's-eye-view parameters
        bev_height = 3.0  # Height of virtual camera in meters
        bev_resolution = 0.01  # BEV resolution in meters/pixel
        bev_size = (1200, 1200)  # BEV image size in pixels

        # Adjust heights relative to ground plane
        bev_points[:, 2] = bev_height - bev_points[:, 2]

        # Remove points below ground (negative height)
        bev_points = bev_points[bev_points[:, 2] > 0]

        # --- Project points onto BEV image plane ---
        u = np.floor(bev_points[:, 0] / bev_resolution + bev_size[0] / 2).astype(np.int32)
        v = np.floor(bev_points[:, 1] / bev_resolution).astype(np.int32)

        # Keep points within image bounds
        valid_idx = (u >= 0) & (u < bev_size[0]) & (v >= 0) & (v < bev_size[1])
        u, v = u[valid_idx], v[valid_idx]
        bev_heights = bev_points[valid_idx, 2]

        # Create BEV depth map
        bev_depth_map = np.zeros(bev_size, dtype=np.float32)
        bev_depth_map[v, u] = bev_heights

        # --- Visualization ---
        plt.figure(figsize=(8, 8))
        plt.imshow(bev_depth_map, cmap='jet', origin='lower')
        plt.colorbar(label='Height (m)')
        plt.title('Bird\'s-eye View Depth Map')
        plt.xlabel('X-axis (pixels)')
        plt.ylabel('Y-axis (pixels)')
        plt.show()


        '''kernel = np.ones((10,10),np.uint8)
        #mask = cv2.erode(mask, kernel, iterations=10)


        import detection.tools.mask as mask_lib
        from scipy.ndimage import distance_transform_edt

        #mask = mask_lib.reduce_mask_width(mask, box_value)

        #mask_left, mask_right, graph = mask_lib.create_side_masks_from_mask(mask, box_value, width=10)

        #mask_compine = np.logical_or(mask_left, mask_right)
        from skimage.morphology import skeletonize
        from scipy.spatial import KDTree
        from skimage.feature import peak_local_max

        imGray = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure it's binary (0 and 255)
        _, road_mask = cv2.threshold(imGray, 127, 255, cv2.THRESH_BINARY)

        # Compute the Euclidean distance transform
        dist_transform = distance_transform_edt(road_mask)

        # Normalize for visualization
        dist_display = (dist_transform / dist_transform.max() * 255).astype(np.uint8)

        # Find local maxima (centerline points)
        centerline_points = peak_local_max(dist_transform, min_distance=5)  # Adjust min_distance for density

        # Convert to NumPy array
        centerline_points = np.array(centerline_points)

        # Find road contour
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        road_contour = contours[0].squeeze()  # Convert to (x, y) format

        # Draw contour
        contour_mask = np.zeros_like(road_mask)
        cv2.drawContours(contour_mask, [road_contour], -1, 255, thickness=2)

        # Build KDTree for fast nearest centerline search
        centerline_kdtree = KDTree(centerline_points)

        left_edge = []
        right_edge = []

        for i in range(len(road_contour)):
            x, y = road_contour[i]

            # Find nearest centerline point
            _, nearest_index = centerline_kdtree.query([y, x])
            nearest_centerline = centerline_points[nearest_index]

            # Compute local gradient from distance transform
            dy, dx = np.gradient(dist_transform)  # Approximate tangent vector
            tangent = np.array([dx[nearest_centerline[0], nearest_centerline[1]],
                                dy[nearest_centerline[0], nearest_centerline[1]]])

            # Get perpendicular normal vector
            normal = np.array([-tangent[1], tangent[0]])  # Rotate 90 degrees

            # Determine left/right using dot product
            vector_to_contour = np.array([y, x]) - nearest_centerline
            side = np.dot(vector_to_contour, normal)

            if side > 0:
                left_edge.append((x, y))
            else:
                right_edge.append((x, y))

        # Convert to NumPy arrays
        left_edge = np.array(left_edge)
        right_edge = np.array(right_edge)

        # Plot results
        plt.imshow(road_mask, cmap='gray')
        plt.scatter(left_edge[:, 0], left_edge[:, 1], c='red', s=2, label="Left Edge")
        plt.scatter(right_edge[:, 0], right_edge[:, 1], c='blue', s=2, label="Right Edge")
        plt.legend()
        plt.title("Final Separated Left and Right Road Boundaries")
        plt.show()'''

        kernel_size = 5

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Dilate and erode mask to get edges
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)

        # Subtract eroded from dilated to get outline
        outline = cv2.absdiff(dilated, eroded)

        plt.imshow(outline, cmap='gray')
        plt.show()


        from birdseye import BirdsEye, CameraConfig


        # Distortion parameters (assuming zero distortion if not calibrated):
        distortion = np.zeros(5)  # [k1, k2, p1, p2, k3]

        # Camera height and angle setup (example):
        camera_config = CameraConfig(
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion,
            height=1.5,  # camera height in meters
            pitch=np.radians(90),  # camera looking straight down (birdseye)
            roll=0,  # assuming camera is level
            yaw=0  # assuming camera facing forward direction
        )

        birdsEye = BirdsEye(camera_config=camera_config)

        birdseye_image = birdsEye.birdseye(mask)

        # Visualization
        plt.imshow(birdseye_image, cmap='gray')
        # plt.gca().invert_yaxis()
        plt.show()


        def next_point(point, direction, step_size):
            """
            Calculates the next 2D point given a current point, a direction vector, and step size.

            Parameters:
                point (array-like): Current point as (x, y).
                direction (np.ndarray): Direction vector as np.array([dx, dy]), values typically between -1 and 1.
                step_size (float): Distance to move along the given direction.

            Returns:
                np.ndarray: Next point coordinates as np.array([x, y]).
            """
            point = np.array(point)

            # Normalize direction vector
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("Direction vector cannot be zero.")
            direction_normalized = direction / norm

            # Calculate next point
            next_pt = point + step_size * direction_normalized

            return next_pt


        def get_normals(direction):
            """
            Computes the perpendicular (normal) vectors of a given 2D direction.

            Parameters:
                direction (np.ndarray): Direction vector [dx, dy].

            Returns:
                tuple: Two normal vectors (rotated +90° and -90°).
            """
            direction = np.asarray(direction)
            normal_1 = np.array([-direction[1], direction[0]])
            normal_2 = np.array([direction[1], -direction[0]])

            return normal_1, normal_2


        def intersect_contour_with_ray(contour, start_point, direction, ray_length=1e5):
            """
            Calculates intersection points between a contour (from OpenCV) and a ray defined by start_point and direction.

            Parameters:
                contour (np.ndarray): Contour array from OpenCV (shape Nx1x2).
                start_point (tuple or np.ndarray): Starting point (x, y) of the ray.
                direction (np.ndarray): Direction vector [dx, dy].
                ray_length (float): Large number to approximate an infinite ray.

            Returns:
                list: Intersection points [(x, y), ...]. Empty if no intersections.
            """
            from shapely.geometry import Polygon, LineString, Point
            # Convert OpenCV contour to shapely polygon
            polygon_points = contour[:, 0, :]  # shape from (N,1,2) to (N,2)
            polygon = Polygon(polygon_points)

            # Create the ray as a long line segment
            direction = direction / np.linalg.norm(direction)
            end_point = start_point + direction * ray_length
            ray = LineString([start_point, end_point])

            print(start_point, end_point)

            # Compute intersection with polygon boundary
            intersection = polygon.boundary.intersection(ray)

            # Process intersection results
            if intersection.is_empty:
                return []
            elif isinstance(intersection, Point):
                return [(intersection.x, intersection.y)]
            elif isinstance(intersection, LineString):
                return list(intersection.coords)
            else:  # MultiPoint or GeometryCollection
                return [(geom.x, geom.y) for geom in intersection.geoms if isinstance(geom, Point)]


        def find_closest_contour_idx(contour, point):
            distances = np.linalg.norm(contour[:, 0, :] - point, axis=1)
            return np.argmin(distances)


        def find_closest_point_in_area(contour, center_idx, external_point, window_size=10):
            num_points = len(contour)
            indices = np.arange(center_idx - window_size, center_idx + window_size + 1)
            indices = indices % num_points  # Handle wrap-around

            area_points = contour[indices, 0, :]
            distances = np.linalg.norm(area_points - external_point, axis=1)
            closest_idx_in_area = np.argmin(distances)

            return area_points[closest_idx_in_area]


        def get_biggest_contour(contours):
            """
            Returns the biggest contour by area from a list of contours.

            Parameters:
                contours (list): List of contours obtained from cv2.findContours.

            Returns:
                np.ndarray: The largest contour.
            """
            if not contours:
                return None

            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour

        def get_centerline_via_normals(mask, bbox, step_size=10):
            mask = mask.astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = get_biggest_contour(contours)
            if len(contours) == 0:
                raise ValueError("No contours found.")
            #polygon = contours[0].squeeze()

            x_min, y_min, x_max, y_max = bbox.astype(int)

            current_point = np.array([(x_min + x_max) // 2, y_max])
            points = [current_point.copy()]

            direction = np.array([0, -1])  # initially upwards

            for i in range(10):
                next_pt = next_point(current_point, direction, step_size)
                normal1, normal2 = get_normals(direction)
                intersection_1 = intersect_contour_with_ray(contour, next_pt, normal1)
                intersection_2 = intersect_contour_with_ray(contour, next_pt, normal2)

                center_idx = find_closest_contour_idx(contour, intersection_1)
                closest_point_1 = find_closest_point_in_area(
                    contour, center_idx, points, window_size=20
                )

                center_idx = find_closest_contour_idx(contour, intersection_2)
                closest_point_2 = find_closest_point_in_area(
                    contour, center_idx, points, window_size=20
                )
                middle_point = (closest_point_1 + closest_point_2) / 2

                print("Middle Points: ", middle_point)

                points.append(middle_point.copy())

                vector = middle_point - current_point
                norm = np.linalg.norm(vector)
                direction = vector / norm

                current_point = middle_point.copy()



            return np.array(points)


        # Example usage
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        centerline_points = get_centerline_via_normals(mask_bin, box_value)

        print("LENGTH CENTER POINTS: ", len(centerline_points))

        # Visualization
        plt.imshow(mask_bin, cmap='gray')
        plt.plot(centerline_points[:, 0], centerline_points[:, 1], 'r-', linewidth=2)
        plt.scatter(centerline_points[:, 0], centerline_points[:, 1], c='yellow')
        #plt.gca().invert_yaxis()
        plt.show()


        ### STEP 3: Remove bottom-left and bottom-right nodes based on real road pixels ###
        '''def remove_bottom_corner_nodes(graph, mask, margin_y=50):
            height, width = mask.shape
            nodes_to_remove = []
            for node, data in graph.nodes(data=True):
                y, x = map(int, data['o'])
                # Only remove nodes that are in the mask (255 region) at bottom-left or bottom-right
                if y > height - margin_y and mask[y, x] == 255:
                    nodes_to_remove.append(node)
            graph.remove_nodes_from(nodes_to_remove)


        # Apply function to remove bottom corner nodes
        remove_bottom_corner_nodes(graph, mask)

        # Car POV (bottom-center point of mask)
        bottom_y = box_value[3]
        bottom_center_x = (box_value[0] + box_value[2]) // 2
        car_point = np.array([bottom_y, bottom_center_x])


        # Add bottom-center node explicitly to graph
        new_node_idx = max(graph.nodes()) + 1
        graph.add_node(new_node_idx, o=car_point)

        # Find closest node ABOVE the bottom-center node
        existing_nodes = np.array([graph.nodes[n]['o'] for n in graph.nodes if n != new_node_idx])
        distances = np.linalg.norm(existing_nodes - np.array([bottom_y, bottom_center_x]), axis=1)
        nearest_node_idx = np.argmin(distances)
        nearest_node_key = list(graph.nodes())[nearest_node_idx]

        # Connect explicitly to nearest node
        graph.add_edge(new_node_idx, nearest_node_idx, weight=1.0, pts=np.array([
            [bottom_y, bottom_center_x],
            graph.nodes[nearest_node_idx]['o']
        ]))'''



        # Plot exactly matching the original image size
        fig, ax = plt.subplots(figsize=(mask.shape[1] / 100, mask.shape[0] / 100), dpi=100)

        # Display original image
        plt.imshow(mask)

        # Plot edges
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green', linewidth=2)

        # Plot nodes
        for node in graph.nodes:
            y, x = graph.nodes[node]['o']
            plt.plot(x, y, 'r.', markersize=8)

        # Adjustments to match exact image dimensions
        plt.axis('off')
        plt.xlim([0, mask.shape[1]])
        plt.ylim([mask.shape[0], 0])  # invert y-axis to match image coordinates exactly
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()


        '''
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
        '''

