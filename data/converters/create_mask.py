import numpy as np
import cv2

def process_image(input_image_path):
    # Read the image
    image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return

    # Create a new image with the same shape as the original image, initialized to black
    new_image = np.zeros_like(image)

    # Define the target color to look for
    target_color = np.array([150, 150, 150])
    target_color_upper = np.array([160, 160, 160])

    mask = cv2.inRange(image, target_color, target_color_upper)

    print(mask)


    # Create a mask where the target color matches
    #mask = np.all(image == target_color, axis=-1)

    print(1 in mask)

    # Set matching pixels to white in the new image
    #new_image[mask] = [255, 255, 255]

    # Display the new image
    cv2.imshow("Processed Image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = '/home/simon/Documents/Master-Thesis/CaSSeD/CaSSed_Dataset_Final/real_world_data/Dataset1_Segmentation/Dataset1B - Powerline/annotations/1562093008.866028224.png'
process_image(input_image_path)
