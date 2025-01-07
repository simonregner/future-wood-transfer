import cv2
import numpy as np

def visualize_polygon_annotation(image_path, annotation_path, class_names):
    """
    Visualize YOLO-style polygon annotations on an image.

    Parameters:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the YOLO annotation file (txt).
        class_names (list): List of class names corresponding to class IDs.

    Returns:
        None: Displays the image with annotations.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Read the annotation file
    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}")

    # Parse the annotation
    image_height, image_width = image.shape[:2]
    for line in lines:
        # Split the line into values
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"Skipping malformed line: {line.strip()}")
            continue

        # First value is the class ID
        class_id = int(parts[0])
        coordinates = list(map(float, parts[1:]))

        # Extract x, y pairs from the coordinates
        points = np.array(
            [(coordinates[i] * image_width, coordinates[i + 1] * image_height) for i in range(0, len(coordinates), 2)],
            dtype=np.int32,
        )

        # Draw the polygon on the image
        color = (0, 255, 0)  # Green for the polygon
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        # Draw the class label near the first point
        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        cv2.putText(image, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow("Polygon Annotations", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Replace with your paths and class names
    image_path = "../../data/GOOSE/images/002638.png"
    annotation_path = "../../data/GOOSE/annotation/002638.txt"
    class_names = [  "background",
            "bugle_road",
            "left_turn",
            "right_turn",
            "road",
            "straight_turn"]  # Replace with your class names

    visualize_polygon_annotation(image_path, annotation_path, class_names)


# Example usage