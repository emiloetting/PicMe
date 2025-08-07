import os
import tqdm
import numpy as np
from colorClusterquantized import *
from skimage.color import rgb2lab, deltaE_ciede2000
import time


cwd = os.getcwd()


# BUILD HISTOGRAMS

# Setup paths
image_data_root = os.path.join(cwd, 'ImageData')
image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Collect hists
ab_hists = []
lightness_hists = []
full_hists_signed = []
full_hists_complete = []

# Get hists
with tqdm.tqdm(total=len(image_paths)) as bar:
    for file_path in image_paths:

        # Calc hists
        hist_ab, hist_l = quantized_image_two_vecs(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        hist_full_signed = quantized_image_signed(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        hist_full_complete = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')

        # Sort hists
        ab_hists.append(hist_ab)
        lightness_hists.append(hist_l)
        full_hists_signed.append(hist_full_signed)
        full_hists_complete.append(hist_full_complete)
        bar.update(1)

# Stack 'em
M_V_ab = np.vstack(ab_hists)
M_V_light = np.vstack(lightness_hists)
M_V_signed = np.stack(full_hists_signed)
M_V_complete = np.stack(full_hists_complete)

# Save hists
np.save(os.path.join(cwd, 'ColorSimilarity3', 'ABHists.npy'), M_V_ab, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity3', 'LightHists.npy'), M_V_light, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity3', 'FullHists_Signed.npy'), M_V_signed, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity3', 'FullHists_Complete.npy'), M_V_complete, allow_pickle=True)

#--------------------------------------------------------------------------------------------------------------------------

# CREATE COST MATRICES FOR EMD

# Get bins in openCV LAB range
center_clrs_lab_ocv = get_bin_centers(l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, round=True).astype(np.uint8)
center_clrs_lab_ocv = center_clrs_lab_ocv.reshape(-1, 1, 3) # reshape

# Interface conversion to rgb for following conversion to correct CIE-LAB
center_clrs_rgb = cv2.cvtColor(center_clrs_lab_ocv, cv2.COLOR_LAB2RGB)

# Conversion to real CIE-LAB
center_clrs_real_lab = rgb2lab(center_clrs_rgb)


# Make explicit col-vector and row-vector to allow for vectorization 
lab1 = center_clrs_real_lab[:, np.newaxis, :]
lab2 = center_clrs_real_lab[np.newaxis, :, :]

# lightness axis
l_centers = np.linspace(0, 100, L_BINS)
l1 = l_centers[:, np.newaxis]  # shape (L_BINS, 1)
l2 = l_centers[np.newaxis, :]  # shape (1, L_BINS)

cost_matrix_light = np.abs(l1 - l2).astype(np.float64)

# Calc full cost-matrix
print("Now calculating cost-matrix using Delta E 2000 metric")
start = time.time()
cost_matrix_color = deltaE_ciede2000(lab1, lab2).astype(np.float64)
cost_matrix_color = np.squeeze(cost_matrix_color)   # get from shape (n,n,1) to (n,n) -> 
cost_matrix_color = cost_matrix_color # Increase cost of different colors to one another , proportionally reduces cost between similar colors


cost_matrix_light = np.abs(l1 - l2).astype(np.float64)

print('Cost mat l shape: ',cost_matrix_light.shape)
end = time.time()
print(f"Computed full cost-matrix in {end-start:.3f} sec.")


# Save matrices
np.save(os.path.join(cwd, 'ColorSimilarity3', 'CostMatrix_full.npy'), cost_matrix_color, allow_pickle=True)
np.save(os.path.join(cwd, "ColorSimilarity3", "CostMatrix_light.npy"), cost_matrix_light, allow_pickle=True)
