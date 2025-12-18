import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import cv2
import json
import cv2
from scipy.ndimage import median_filter
import sys


def build_rgb_kdtree(csv_path='rgb_stress_lut.csv'):
    lut_df = pd.read_csv(csv_path)
    stacked_rgb = lut_df[['R', 'G', 'B']].values
    stress_values = lut_df['Stress'].values
    kdtree = KDTree(stacked_rgb)
    return kdtree, stress_values

def build_CIE_kdtree(csv_path=r".\Stress_CIE_xy.csv"):
    lut_df = pd.read_csv(csv_path)
    stacked_rgb = lut_df[['CIE_x', 'CIE_y']].values
    stress_values = lut_df['Stress'].values
    kdtree = KDTree(stacked_rgb)
    return kdtree, stress_values


def build_dual_CIE_kdtrees(csv_path, roi_min, roi_max, left_min, left_max, right_max):
    lut_df = pd.read_csv(csv_path)
    xy = lut_df[['CIE_x', 'CIE_y']].values
    stress = lut_df['Stress'].values

    roi_mask = (stress >= roi_min) & (stress <= roi_max)
    left_middle_mask = (stress >= left_min) & (stress <= left_max)
    right_mask = stress <= right_max
    left_top_mask = (stress >= left_min) & (stress < left_max)

    kdtree_roi = KDTree(xy[roi_mask])
    stress_roi = stress[roi_mask]

    kdtree_left_middle = KDTree(xy[left_middle_mask])
    stress_left_middle = stress[left_middle_mask]

    kdtree_right = KDTree(xy[right_mask])
    stress_right = stress[right_mask]

    kdtree_left_top = KDTree(xy[left_top_mask])
    stress_left_top = stress[left_top_mask]

    return (kdtree_roi, stress_roi), (kdtree_left_middle, stress_left_middle), (kdtree_right, stress_right), (kdtree_left_top, stress_left_top)


def compute_stress_map_from_rgb(img_file, image, kdtree_groups, mask, background_threshold=0.05):
    h, w, _ = image.shape
    cie_pixels = image.reshape(-1, 3)
    xys = cie_pixels[:, :2]
    Y_channel = cie_pixels[:, 2]

    (kdtree_roi, stress_roi),\
    (kdtree_left_middle, stress_left_middle), \
    (kdtree_right, stress_right), \
    (kdtree_left_top, stress_left_top) = kdtree_groups

    is_background = Y_channel < background_threshold

    stress_map_flat = np.full((h * w,), np.nan)

    x1, y1 = mask["top_left"]
    x2, y2 = mask["bottom_right"]
    

    for idx in range(h * w):
        if is_background[idx]:
            continue

        i = idx // w
        j = idx % w
        query_xy = xys[idx]

        if j < x2:
            if x1 <= j < x2 and y1 <= i < y2:
                _, index = kdtree_roi.query(query_xy)
                stress = stress_roi[index]
            elif (i < h //3 - 15 or i >= 2 * h // 3 + 15):
                _, index = kdtree_left_top.query(query_xy)
                stress = stress_left_top[index]
            else:
                _, index = kdtree_left_middle.query(query_xy)
                stress = stress_left_middle[index]
        else:
            _, index = kdtree_right.query(query_xy)
            stress = stress_right[index]

        stress_map_flat[idx] = stress

    return stress_map_flat.reshape(h, w)


def img_to_cie_xyY_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    
    def gamma_correct(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r_lin = gamma_correct(img_rgb[..., 0])
    g_lin = gamma_correct(img_rgb[..., 1])
    b_lin = gamma_correct(img_rgb[..., 2])

    X = r_lin * 0.4124 + g_lin * 0.3576 + b_lin * 0.1805
    Y = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722
    Z = r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505

    denom = X + Y + Z
    x = np.where(denom == 0, 0, X / denom)
    y = np.where(denom == 0, 0, Y / denom)

    img_cie_xyY = np.stack((x, y, Y), axis=-1)

    return img_cie_xyY


def horizontal_anomaly_correction(stress_map, mask, delta_threshold, num_neighbors):

    H, W = stress_map.shape
    stress_fixed = stress_map.copy()
    
    segment_width = W // 3

    stride = segment_width // 5

    for i in range(H):
        row = stress_map[i]
        valid_mask = ~np.isnan(row)
        valid_ratio = np.sum(valid_mask) / W

        if valid_ratio < 0.7:
            continue  

        for seg in range(0, W, stride):
            start = seg
            end = min(start + stride, W)

            segment = row[start:end]
            segment_valid = segment[~np.isnan(segment)]

            if len(segment_valid) == 0:
                continue

            segment_mean = np.mean(segment_valid) 

            for j in range(start, end):
                val = row[j]
                if np.isnan(val):
                    continue

                x1, y1 = mask["top_left"]
                x2, y2 = mask["bottom_right"]
                if x1 - 30 <= j < x2 + 30 and y1 - 20 <= i < y2 + 20:
                    continue

                if np.abs(val - segment_mean) > delta_threshold:
                    neighbors = []

                    offset = 1
                    while len(neighbors) < num_neighbors and (j - offset >= 0 or j + offset < W):
                        strict_delta_threshold = delta_threshold / 2

                        if j - offset >= 0 and not np.isnan(row[j - offset]) and np.abs(row[j - offset] - segment_mean) <= strict_delta_threshold: 
                            neighbors.append(row[j - offset])
                        if j + offset < W and not np.isnan(row[j + offset]) and np.abs(row[j + offset] - segment_mean) <= strict_delta_threshold:
                            neighbors.append(row[j + offset])
                        if i - offset >= 0 and not np.isnan(stress_map[i - offset, j]) and np.abs(stress_map[i - offset, j] - segment_mean) <= strict_delta_threshold:
                            neighbors.append(stress_map[i - offset, j])
                        if i + offset < H and not np.isnan(stress_map[i + offset, j]) and np.abs(stress_map[i + offset, j] - segment_mean) <= strict_delta_threshold:
                            neighbors.append(stress_map[i + offset, j])
                        offset += 1
                        
                    if neighbors:
                        stress_fixed[i, j] = np.mean(neighbors)

    return stress_fixed


def process_folder(input_folder, output_folder, kdtree_groups, roi_dict):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    

    for idx, img_file in enumerate(image_files):
        if idx < 2:
            group_idx = 0 
        elif 2 <= idx <= 7:
            group_idx = 1
        else:
            group_idx = 2

        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        img = img_to_cie_xyY_tensor(img)


        mask_roi = roi_dict.get(img_path)

        stress_map = compute_stress_map_from_rgb(img_file, img, kdtree_groups[group_idx], mask_roi)

        stress_map = horizontal_anomaly_correction(stress_map, mask_roi, delta_threshold = 4, num_neighbors = 10) 

        stress_map_5 = median_filter(stress_map, size=(1, 5))
        stress_map_5 = median_filter(stress_map_5, size=(5, 1))

        output_path_raw = os.path.join(output_folder, img_file.replace('.png', '_stress.png'))
        plt.figure(figsize=(8, 6))
        im = plt.imshow(stress_map_5, cmap="jet", interpolation="nearest", vmin=0, vmax=20)
        cbar = plt.colorbar(im, label="Stress (MPa)")
        cbar.set_ticks([0, 5, 10, 15, 20])
        plt.axis("off")
        plt.savefig(output_path_raw, bbox_inches='tight', pad_inches=0)
        plt.close()
    
        print(f"Processed {img_file} (raw and corrected)")


def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size), scale


if __name__ == "__main__":

    project_root = r".\StressField_code"

    input_folder_list = [
    r".\StressField_code\top"
    ]

    output_folder_list = [
    r".\StressField_code\StressField"
    ]

    roi_path = os.path.join(project_root, "selected_rois.json")

    choice = input("Should all ROIs be reselected? (y/n):").strip().lower()

    roi_dict = {}

    if choice == "n" and os.path.exists(roi_path):
        with open(roi_path, "r") as f:
            roi_dict = json.load(f)
        print("Existing ROI data has been read.")
    else:
        roi_dict = {} 
        for i, input_folder in enumerate(input_folder_list):
            output_folder = output_folder_list[i]

            if not os.path.exists(input_folder):
                print(f"The folder does not exist: {input_folder}")
                sys.exit(1)

            os.makedirs(output_folder, exist_ok=True)

            img_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])  

            for img_file in img_files:
                img_path = os.path.join(input_folder, img_file)

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Unable to read image: {img_path}")
                    continue
                
                resized_img, scale = resize_for_display(img)
                roi = cv2.selectROI("Select ROI", resized_img, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ROI")

                x, y, w, h = roi
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                roi_dict[img_path] = {
                    "top_left": top_left,
                    "bottom_right": bottom_right
                }

        with open(roi_path, "w") as f:
            json.dump(roi_dict, f, indent=4)
        print(f"All ROIs have been saved to {roi_path}")

    codebook_path = os.path.join(project_root, "Stress_CIE_xy.csv")

    kdtree_groups = [
        build_dual_CIE_kdtrees(codebook_path, 2, 14, 0, 10, 8), 
        build_dual_CIE_kdtrees(codebook_path, 12.7, 20, 2, 12, 9),
        build_dual_CIE_kdtrees(codebook_path, 12.7, 20, 2, 11, 11)
    ]


    codebook_path = r".\StressField_code\Stress_CIE_xy.csv"
    kdtree, stress_values = build_CIE_kdtree(codebook_path)
    
    for i, input_folder in enumerate(input_folder_list):
        output_folder = output_folder_list[i]

        tif_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')]) 

        process_folder(input_folder, output_folder, kdtree_groups, roi_dict)


