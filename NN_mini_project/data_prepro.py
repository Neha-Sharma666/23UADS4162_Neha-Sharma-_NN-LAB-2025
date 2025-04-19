import os
import cv2
import numpy as np

# Corrected Paths
input_base = r"C:\Users\NEHA\OneDrive\Desktop\NN_project_Datset"
output_base = r"C:\Users\NEHA\OneDrive\Desktop\NN_pro_preprosecced_dataset"

# Create output base folder if it doesn't exist
os.makedirs(output_base, exist_ok=True)

# Define rotation angles and two contrast factors
rotation_angles = [0, 90, 180]
contrast_factors = [1.0, 1.3]  # original contrast + 1 new contrast

# Loop over each species folder
for species_folder in os.listdir(input_base):
    species_input_path = os.path.join(input_base, species_folder)
    species_output_path = os.path.join(output_base, species_folder)

    if not os.path.isdir(species_input_path):
        continue  # Skip if not a folder

    os.makedirs(species_output_path, exist_ok=True)

    # Loop through each image in the species folder
    for img_file in os.listdir(species_input_path):
        img_path = os.path.join(species_input_path, img_file)
        img_name = os.path.splitext(img_file)[0]

        # Read as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        # Resize to 256x256
        image_resized = cv2.resize(image, (256, 256))

        # Apply adaptive Gaussian thresholding
        image_thresh = cv2.adaptiveThreshold(
            image_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Apply all rotations and both contrast levels
        for angle in rotation_angles:
            if angle == 0:
                rotated = image_thresh.copy()
            elif angle == 90:
                rotated = cv2.rotate(image_thresh, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image_thresh, cv2.ROTATE_180)

            for contrast in contrast_factors:
                adjusted = cv2.convertScaleAbs(rotated, alpha=contrast, beta=0)
                aug_filename = f"{img_name}_rot{angle}_c{str(contrast).replace('.', '')}.jpg"
                aug_path = os.path.join(species_output_path, aug_filename)
                cv2.imwrite(aug_path, adjusted)

print(" Thresholding + 3 Rotations × 2 Contrast Levels (Total 6 images per input image).")