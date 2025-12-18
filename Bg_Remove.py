import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import label
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

def capture_video_frames(video_file, interval, output_folder):
    """
    Read the video file and capture frames at the specified interval, saving them as TIFF files.
    """
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval)  # Capture a frame every interval seconds

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
 
 # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    image_count = 0

    # Create a tqdm progress bar
    with tqdm(total=total_frames, desc="Capturing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                image_name = os.path.join(output_folder, f"frame_{image_count}.jpg")
                cv2.imwrite(image_name, frame)
                image_count += 1

            frame_count += 1
            pbar.update(1)  # Update the progress bar

    cap.release()
    print(f"Video frames captured and saved in {output_folder}")


def crop_and_remove_black_background(input_folder):
    """
    Process all captured frames: crop the central 1/3, remove black background, retain the largest continuous region.
    """
    # Set a threshold to determine whether a pixel is nearly black
    threshold = 50  # can be adjusted

    # Get the list of image paths
    image_paths = list(Path(input_folder).rglob('*.tif'))

    # Create a tqdm progress bar
    with tqdm(total=len(image_paths), desc="Processing images") as pbar:
        for image_path in image_paths:
            # open the image
            img = Image.open(image_path)
            width, height = img.size

            # Crop the central 1/3 of the width.
            left = width // 3
            right = 2 * (width // 3)
            cropped_img = img.crop((left, 0, right, height))

            # Convert to RGBA mode to support a transparent background
            cropped_img = cropped_img.convert("RGBA")

            # Retrieve the pixel data of the image
            data = np.array(cropped_img)
            r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

            # Set the nearly black regions to be transparent using the threshold
            black_pixels = (r < threshold) & (g < threshold) & (b < threshold)
            data[black_pixels] = [0, 0, 0, 0]

            # Convert to a grayscale image for connected component labeling
            gray = cv2.cvtColor(data, cv2.COLOR_RGBA2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Use SciPy to find connected regions
            labeled_array, num_features = label(binary)

            # Find the largest connected region
            max_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1  # Find the label of the largest connected region
            max_region = (labeled_array == max_label).astype(np.uint8) * 255  # The binary image of the largest connected region
            # Make non-maximum regions transparent
            max_region_4d = np.repeat(max_region[:, :, np.newaxis], 4, axis=2)  # Expand the max_region to 4D
            data[max_region == 0] = [0, 0, 0, 0]  # Make non-maximum regions transparent

            # Save the processed image back to its original location
            final_img = Image.fromarray(data)
            final_img.save(image_path)

            pbar.update(1)  # Update the progress bar
    print("All images processed.")

# Function to open file dialog for selecting folder or file
def select_folder():
    Tk().withdraw()  # Hide the root Tk window
    folder_path = askdirectory(title="Select Folder")
    return folder_path

def select_file():
    Tk().withdraw()  # Hide the root Tk window
    file_path = askopenfilename(title="Select File", filetypes=(("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")))
    return file_path

if __name__ == "__main__":
    # User enter video file path and interval
    # User select stress-strain data CSV file path
    video_file = select_file()  # Open file dialog to select file
    # video_file = input("Please enter the video file path: ")
    interval = float(input("Please enter the interval(sec): "))

    # temporary folder path
    temp_folder = r".\temporary"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # step1: capture video frames
    capture_video_frames(video_file, interval, temp_folder)

    # step2: process the image, cut the middle and remove the black background
    #crop_and_remove_black_background(temp_folder)
