import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


source_folder = r".\StressField_code"

base_output = r".\StressField_code"

output_folders = [
    os.path.join(base_output, "test_top"),
    os.path.join(base_output, "test_middle"),
    os.path.join(base_output, "test_bottom")
]

for folder in output_folders:
    os.makedirs(folder, exist_ok=True)


def extract_number(filename):
    basename = os.path.splitext(filename)[0]
    digits = ''.join(filter(str.isdigit, basename))
    return int(digits) if digits else 0

jpg_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
jpg_files.sort(key=extract_number)


y_top_list, y_bottom_list, x_left_list, x_right_list = [], [], [], []

res_y, res_s1, res_s2, res_s3 = [], [], [], []


jpg_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')],
                   key=extract_number)

top_length, middle_length, bottom_length = [], [], []


for img_name in tqdm(jpg_files):

    img_path = os.path.join(source_folder, img_name)

    img = cv2.imread(img_path)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    row_means = np.mean(gray, axis=1)

    foreground_row_mask = row_means > 15  

    y_nonzero = np.where(foreground_row_mask)[0]


    if len(y_nonzero) < 10:
        print(f"{img_name}: Skip if no sample area detected")
        continue
    y_top, y_bottom = y_nonzero[0], y_nonzero[-1]


    split_indices = np.where(np.diff(y_nonzero) > (y_bottom - y_top) / 8)[0]


    group_boundaries = []
    prev = 0
    for idx in split_indices:
        group = y_nonzero[prev:idx + 1]
        if len(group) > 0:
            group_boundaries.append((group[0], group[-1]))
        prev = idx + 1

    group = y_nonzero[prev:]
    if len(group) > 0:
        group_boundaries.append((group[0], group[-1]))

    res_s1.append(group_boundaries[0][1] - group_boundaries[0][0])
    res_s2.append(group_boundaries[1][1] - group_boundaries[1][0])
    res_s3.append(group_boundaries[2][1] - group_boundaries[2][0])

    col_means = np.mean(gray, axis=0)

    foreground_col_mask = col_means > 30

    x_nonzero = np.where(foreground_col_mask)[0]
    if len(x_nonzero) < 10:
        print(f"{img_name}: Skip if no sample area detected")
        continue
    x_left, x_right = x_nonzero[0], x_nonzero[-1]

    print(x_left, x_right)

    split_indices_x = np.where(np.diff(x_nonzero) > (x_right - x_left) / 12)[0]

    col_boundaries = []
    prev = 0
    for idx in split_indices_x:
        group = x_nonzero[prev:idx + 1]
        if len(group) > 0:
            col_boundaries.append((group[0], group[-1]))
        prev = idx + 1

    group = x_nonzero[prev:]
    if len(group) > 0:
        col_boundaries.append((group[0], group[-1]))

    x_left, x_right = col_boundaries[-1]


    res_y.append(y_bottom - y_top)
    y_top_list.append(y_top)
    y_bottom_list.append(y_bottom)
    x_left_list.append(x_left)
    x_right_list.append(x_right)

    specimen_region = rotated[y_top:y_bottom, :, :]

    h = specimen_region.shape[0]
    top = rotated[group_boundaries[0][0]:group_boundaries[0][1]+50, x_left-20: x_right+50, :]
    middle = rotated[group_boundaries[1][0]:group_boundaries[1][1]+50, x_left-20: x_right+50, :]
    bottom = rotated[group_boundaries[2][0]:group_boundaries[2][1]+50, x_left-20: x_right+50, :]


    base_filename = os.path.splitext(img_name)[0]
    cv2.imwrite(os.path.join(output_folders[0], f"{base_filename}_top.png"), top)
    cv2.imwrite(os.path.join(output_folders[1], f"{base_filename}_middle.png"), middle)
    cv2.imwrite(os.path.join(output_folders[2], f"{base_filename}_bottom.png"), bottom)

    print(f"Processing complete: {img_path}")


