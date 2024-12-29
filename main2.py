import os
import cv2
from nudenet import NudeDetector

# List of classes to apply the blur
valid_classes = [
    'FACE_FEMALE',
    'FEMALE_GENITALIA_COVERED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'ANUS_COVERED'
]

# Ask for the folder containing the images
input_folder = input("Enter the folder containing the images: ")
output_folder = input("Enter the folder where the processed images will be saved: ")

# Ensure the output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all the image files in the input folder
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # List of valid image extensions
images = [
    os.path.join(input_folder, f) for f in os.listdir(input_folder) 
    if os.path.splitext(f)[1].lower() in image_extensions
]

# Initialize the NudeDetector
detector = NudeDetector()

# Process each image in the list
for image_path in images:
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image {image_path}")
        continue

    height, width = image.shape[:2]

    # Detect objects in the image
    iaResults = detector.detect(image_path)

    for detection in iaResults:
        # Check if the class matches any of the valid classes
        if detection['class'] in valid_classes:
            x, y, w, h = detection['box']

            # Calculate the coordinates for the ROI
            x2 = x + w
            y2 = y + h

            # Ensure coordinates are within bounds
            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # Ensure x1 < x2 and y1 < y2
            if x1 >= x2 or y1 >= y2:
                continue

            # Define the region of interest (ROI) from the coordinates
            roi = image[y1:y2, x1:x2]

            # Apply a strong Gaussian blur to the ROI
            blurred_roi = cv2.GaussianBlur(roi, (1001, 1001), 200)

            # Replace the original ROI with the blurred ROI in the image
            image[y1:y2, x1:x2] = blurred_roi

    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '_blurred.jpg'))
    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")
