import os
import cv2
from nudenet import NudeDetector
import tkinter as tk
from tkinter import filedialog, messagebox

# List of available classes for blurring
all_classes = [
    'FACE_FEMALE',
    'FEMALE_GENITALIA_COVERED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'ANUS_COVERED'
]

# Function to open a file dialog and select a folder
def ask_for_folder(title="Select a folder"):
    folder = filedialog.askdirectory(title=title)
    if not folder:
        messagebox.showerror("Error", "No folder selected!")
    return folder

# Function to create a checkbox UI for selecting classes
def create_class_selection_ui():
    root = tk.Tk()
    root.title("Select Classes to Blur")
    
    selected_classes = []

    def toggle_class(cls):
        if cls in selected_classes:
            selected_classes.remove(cls)
        else:
            selected_classes.append(cls)

    # Create checkboxes for each class
    checkboxes = []
    for cls in all_classes:
        var = tk.BooleanVar()
        checkboxes.append(var)
        checkbox = tk.Checkbutton(root, text=cls, variable=var, command=lambda c=cls: toggle_class(c))
        checkbox.pack(anchor="w")

    # Create a Submit Button
    def submit():
        root.quit()

    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack()

    root.mainloop()

    return selected_classes

# Create the GUI for selecting the folders and classes
input_folder = ask_for_folder("Select the input folder containing images")
if not input_folder:
    exit()

output_folder = ask_for_folder("Select the output folder")
if not output_folder:
    exit()

valid_classes = create_class_selection_ui()

if not valid_classes:
    messagebox.showerror("Error", "No classes selected!")
    exit()

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
        # Check if the class matches any of the selected classes
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
