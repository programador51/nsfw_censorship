from nudenet import NudeDetector
import cv2
import numpy as np

# List of classes to apply the blur
valid_classes = [
    'FACE_FEMALE',
    'FEMALE_GENITALIA_COVERED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'ANUS_COVERED'
]

images = [
    'C:/Users/programador51/Pictures/test/3078993569.jpg',
    'C:/Users/programador51/Pictures/test/3078993571.jpg',
    'C:/Users/programador51/Pictures/test/3078993573.jpg']

detector = NudeDetector()
# the 320n model included with the package will be used
# data = detector.censor('image_1.jpg',["FACE_FEMALE","FEMALE_GENITALIA_COVERED","FEMALE_BREAST_EXPOSED","FEMALE_GENITALIA_EXPOSED","ANUS_COVERED"]) # returns censored image output path
image = cv2.imread(images[0])

height, width = image.shape[:2]
# returns censored image output path
iaResults = detector.detect(images[0])


for detection in iaResults:
    # Check if the class matches any of the valid classes
    if detection['class'] in valid_classes:
        # Extract the bounding box coordinates (x1, x2, y1, y2)
        x, y, w, h = detection['box']

        x2 = x + w
        y2 = y + h

        # Ensure that the coordinates are within image bounds
        x1 = max(0, min(x, width - 1))
        y1 = max(0, min(y, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Ensure that x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            continue

        # Define the region of interest (ROI) from the coordinates
        roi = image[y1:y2, x1:x2]

        # Apply a Gaussian blur to the ROI (you can adjust the kernel size as needed)
        blurred_roi = cv2.GaussianBlur(roi, (501, 501), 100)

        # Replace the original ROI with the blurred ROI in the image
        image[y1:y2, x1:x2] = blurred_roi

cv2.imwrite('output_image_blurred.jpg', image)
print("Image saved as 'output_image_blurred.jpg'")
