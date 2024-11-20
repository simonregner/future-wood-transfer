import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def extract_colors(image_path, num_colors=5):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image at '{image_path}'. Please check the file path.")
        return None

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Use KMeans to find the dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the colors and percentage distribution
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = np.bincount(labels)

    # Convert colors to a 0-1 range
    colors_normalized = colors / 255.0

    # Convert colors to hex format for labels
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2])) for color in colors]

    # Plot the colors
    plt.figure(figsize=(12, 6))
    wedges, texts = plt.pie(label_counts, colors=[tuple(color) for color in colors_normalized], startangle=90)

    # Add color labels to each wedge
    for i, wedge in enumerate(wedges):
        # Calculate angle for label placement
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.1 * np.cos(np.deg2rad(angle))
        y = 1.1 * np.sin(np.deg2rad(angle))
        plt.text(x, y, hex_colors[i], color='black', ha='center', va='center', fontsize=12)

    plt.show()

    return colors

image_path = '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1_Segmentation/Dataset1B - Powerline/annotations/1562093032.465825280.png'
colors = extract_colors(image_path)

if colors is not None:
    print("Extracted Colors (in RGB):", colors)