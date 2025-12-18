import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.colors import LinearSegmentedColormap
import glob
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import re


# Set input folder path and output folder path
input_folder = r".\temporary"
output_folder = r".\temporary\output"
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exist

# Get all TIFF images in the folder
# image_files = sorted(glob.glob(f"{input_folder}/*.jpg"), key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]))

def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf') 

image_files = sorted(glob.glob(f"{input_folder}/*.jpg"), key=extract_number_from_filename)


def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf') 

image_files = sorted(glob.glob(f"{input_folder}/*.jpg"), key=extract_number_from_filename)

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read {img_path}, skipping rotation and cropping.")
        continue

    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = rotated_img.shape[:2]

    x1 = w // 3
    x2 = 2 * w // 3

    y1 = h // 3
    y2 = h

    cropped_img = rotated_img[y1:y2, x1:x2]

    cv2.imwrite(img_path, cropped_img)


# Create an Excel
excel_file = os.path.join(output_folder, "sampling_data.xlsx")
df = pd.DataFrame(columns=["Frame Number", "Average R", "Average G", "Average B"])

# Save RGB value
frame_numbers = []
r_values = []
g_values = []
b_values = []

# Process every TIF image with progress bar
for index, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images", ncols=100):
    # print(index, image_file)
    # print('='*100)
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read the full TIF image
    
    if img is None:
        print(f"Error: Cannot read {image_file}, skipping.")
        continue

    # Process RGBA images (if there is a transparent channel)
    if img.shape[-1] == 4:  # Check if it is 4-channel (RGBA)
        # alpha_channel = img[:, :, 3]  # Extract Alpha channel
        # mask = alpha_channel > 0  # Generates a mask for an opaque area
        mask = np.any(img[:, :, :3] > 20, axis=-1)  # Filter out the black background

    else:  # If the image has 3 channels (RGB)，assume the black background is (0,0,0)
        mask = np.any(img[:, :, :3] > 10, axis=-1)  # Filter out the black background

    # Gets coordinates for all non-transparent/non-black areas
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        print(f"Warning: No valid pixels found in {image_file}, skipping.")
        continue
    
    center_y, center_x = np.median(coords, axis=0).astype(int)

    # Define sample area（sidelength L/2 pixel）
    L = 10
    x1 = max(center_x - L, 0)
    y1 = max(center_y - L, 0)
    x2 = min(center_x + L, img.shape[1])
    y2 = min(center_y + L, img.shape[0])

    # Extraction sampling area
    sample_region = img[y1:y2, x1:x2, :3]  # only RGB channels

    # Calculates the average RGB value for the sample area
    avg_r = np.mean(sample_region[:, :, 2])  # red channel
    avg_g = np.mean(sample_region[:, :, 1])  # green channel
    avg_b = np.mean(sample_region[:, :, 0])  # blue channel

    # Save data
    frame_numbers.append(index + 1)
    r_values.append(avg_r)
    g_values.append(avg_g)
    b_values.append(avg_b)

    # Mark the sampling area on the image (red border)
    marked_img = img[:, :, :3].copy()  # only RGB part
    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red border

    # Save the marked image (TIF format)
    output_image_path = os.path.join(output_folder, f"marked_{image_file}")
    cv2.imwrite(output_image_path, marked_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # TIF saved

# save data to Excel
df["Frame Number"] = frame_numbers
df["Average R"] = r_values
df["Average G"] = g_values
df["Average B"] = b_values
df.to_excel(excel_file, index=False)

plt.figure(figsize=(10, 6))

plt.scatter(frame_numbers, r_values, label='Red', color='red', s=10, alpha=0.7)
plt.scatter(frame_numbers, g_values, label='Green', color='green', s=10, alpha=0.7)
plt.scatter(frame_numbers, b_values, label='Blue', color='blue', s=10, alpha=0.7)

plt.xlabel("Frame Number (Second)")
plt.ylabel("Average RGB Value")
plt.title("RGB Value Trends Over Time")
plt.legend()
plt.grid(True)

# Save and show the plot
graph_image_path = os.path.join(output_folder, "rgb_trend.png")
plt.savefig(graph_image_path)
plt.show()

print(f"Data saved to {excel_file}")
print(f"Graph saved to {graph_image_path}")
print(f"Marked images saved to {output_folder}")


# Import stress-strain data
# Time(s) = interval(s) * Num
# User enter time interval
interval = float(input("Please enter the interval time for video frames Interval (s): "))

# Function to open file dialog for selecting folder or file
def select_folder():
    Tk().withdraw()  # Hide the root Tk window
    folder_path = askdirectory(title="Select Folder")
    return folder_path

def select_file():
    Tk().withdraw()  # Hide the root Tk window
    file_path = askopenfilename(title="Select File", filetypes=(("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")))
    return file_path

# User select stress-strain data CSV file path
stress_strain_file = select_file()  # Open file dialog to select file

# # User enter stress-strain data CSV file path
# stress_strain_file = input("Please enter test data Excel file path: ")

# Read the CSV file and display column information
df_stress = pd.read_csv(stress_strain_file)
columns_list = df_stress.columns.tolist()

print("\nThe column names of the file are as follows (starting from number 1): ")
for i, col_name in enumerate(columns_list, 1):
    print(f"{i}: {col_name}")

# User input the column numbers for stress and strain data (starting from 1)
time_col_index = int(input("Please enter the column number corresponding to Time(s): ")) - 1
disp_col_index = int(input("Please enter the column number corresponding to Displacement(mm): ")) - 1
load_col_index = int(input("Please enter the column number corresponding to Load(N): ")) - 1

# Get the column names selected by the user
time_col = columns_list[time_col_index]
disp_col = columns_list[disp_col_index]
load_col = columns_list[load_col_index]

# User enter the dimension of the test sample
L = float(input("Please enter sample Length L (mm): "))  
W = float(input("Please enter sample Width W (mm): "))  
H = float(input("Please enter sample Thickness H (mm): "))  

# Calculate stress-strain data
df_stress["Strain"] = df_stress.iloc[:, disp_col_index] / L
df_stress["Stress (MPa)"] = df_stress.iloc[:, load_col_index] / (W * H)

# Read previous RGB data
input_folder = r".\temporary\output"
rgb_file = os.path.join(input_folder, "sampling_data.xlsx")
df_rgb = pd.read_excel(rgb_file)

# Calculate Time(s) = Frame Number * interval
df_rgb["Time(s)"] = df_rgb["Frame Number"] * interval

# Match RGB and Stress-Strain data based on the closest Time(s)
def find_closest_time(row, stress_data, time_col):
    closest_index = (stress_data[time_col] - row['Time(s)']).abs().argmin()
    closest_time = stress_data.iloc[closest_index][time_col]
    closest_strain = stress_data.iloc[closest_index]["Strain"]
    closest_stress = stress_data.iloc[closest_index]["Stress (MPa)"]
    return pd.Series([closest_strain, closest_stress], index=["Strain", "Stress (MPa)"])

# Apply function to match closest times
df_rgb[["Strain", "Stress (MPa)"]] = df_rgb.apply(find_closest_time, axis=1, stress_data=df_stress, time_col=time_col)
print(f"After apply, df_rgb shape: {df_rgb.shape}")
print(f"After apply, df_rgb head:\n{df_rgb.head()}")

# Save to a new Excel file
rgb_file1 = os.path.join(input_folder, "stress-RGB_data_test.xlsx")
df_rgb.to_excel(rgb_file1, index=False, engine='openpyxl')
print(f"\nData has been saved to: {rgb_file1}")


# # If df_rgb has been defined
# df_rgb = df_rgb.copy()  # Copy the original data to avoid modifying the original data

# # Process Stress (MPa) decreasing issue
# for i in range(1, len(df_rgb)):
#     if df_rgb.loc[i, "Stress (MPa)"] > 30 and df_rgb.loc[i, "Stress (MPa)"] < 0.99 * df_rgb.loc[i - 1, "Stress (MPa)"]:  # plastic stress ~ 45 MPa
#         df_rgb = df_rgb.iloc[:i]  # Truncate, discarding the current index and all rows after it
#         break
# print(df_rgb)

# # Overwrite the original Excel file
# rgb_file1 = os.path.join(input_folder, "stress-RGB_data_test.xlsx")
# df_rgb.to_excel(rgb_file1, index=False)


# If df_rgb is defined and contains "Stress (MPa)"、"Average R"、"Average G"、"Average B"
stress_values = df_rgb["Stress (MPa)"].values
r_values = df_rgb["Average R"].values
g_values = df_rgb["Average G"].values
b_values = df_rgb["Average B"].values

# Normalize RGB color value
r_values = r_values / 255.0
g_values = g_values / 255.0
b_values = b_values / 255.0

# Generate an array of plot background color
bg_colors = np.column_stack([r_values, g_values, b_values])

# Create background color
fig, ax = plt.subplots(figsize=(10, 6))
extent = [min(stress_values), max(stress_values), 0, 255]
ax.imshow([bg_colors], aspect="auto", extent=extent, origin="lower")

# Plot the RGB curve
ax.scatter(stress_values, df_rgb["Average R"], color="red", label="Red", s=10)
ax.scatter(stress_values, df_rgb["Average G"], color="green", label="Green", s=10)
ax.scatter(stress_values, df_rgb["Average B"], color="blue", label="Blue", s=10)

# Set axis labels and title
plt.xlabel("Stress (MPa)")
plt.ylabel("RGB Value")
plt.title("Stress-RGB Relationship with Background Color")
ax.legend()

# Save Figure before show
output_image = os.path.join(input_folder, "stress_RGB_plot.png")
plt.savefig(output_image)
print(f"Image has been saved to: {output_image}")

# Then show the plot
plt.show()