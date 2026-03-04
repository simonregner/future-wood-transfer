from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

# Define the ontology: mapping text prompts to class labels
ontology = CaptionOntology(
    {
        "road": "road",
        #"trees": "trees",
        # Add more mappings as needed
    }
)

# Initialize the Grounded SAM 2 model with the ontology
base_model = GroundedSAM2(ontology=ontology)

from autodistill.utils import plot
import cv2

# Path to a test image
test_image_path = "test_images/rgb_1744035922782076669.png"

# Run inference on the test image
results = base_model.predict(test_image_path)

# Visualize the results
image = cv2.imread(test_image_path)
plot(image=image, classes=base_model.ontology.classes(), detections=results)