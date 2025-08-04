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
pos_hists = []
neg_hists = []
lightness = []
full_hists_signed = []
full_hists_complete = []

# Get hists
with tqdm.tqdm(total=len(image_paths)) as bar:
    for file_path in image_paths:

        # Calc hists
        hist_pos, hist_neg, l_hist = quantized_image_three_vecs(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        hist_full_signed = quantized_image_signed(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        hist_full_complete = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')

        # Sort hists
        pos_hists.append(hist_pos)
        neg_hists.append(hist_neg)
        lightness.append(l_hist)
        full_hists_signed.append(hist_full_signed)
        full_hists_complete.append(hist_full_complete)
        bar.update(1)

# Stack 'em
M_V_pos = np.vstack(pos_hists)
M_V_neg = np.vstack(neg_hists)
M_V_light = np.vstack(lightness)
M_V_signed = np.stack(full_hists_signed)
M_V_complete = np.stack(full_hists_complete)

# Save hists
np.save(os.path.join(cwd, 'ColorSimilarity2', 'PosHists.npy'), M_V_pos, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity2', 'NegHists.npy'), M_V_neg, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity2', 'LightHists.npy'), M_V_light, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity2', 'FullHists_Signed.npy'), M_V_signed, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity2', 'FullHists_Complete.npy'), M_V_complete, allow_pickle=True)

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

# Calc full cost-matrix
print("Now calculating cost-matrix using Delta E 2000 metric")
start = time.time()
cost_matrix_color = deltaE_ciede2000(lab1, lab2).astype(np.float64)
cost_matrix_color = np.squeeze(cost_matrix_color)   # get from shape (n,n,1) to (n,n) -> 
cost_matrix_color = cost_matrix_color # Increase cost of different colors to one another , proportionally reduces cost between similar colors
end = time.time()
print(f"Computed full cost-matrix in {end-start:.3f} sec.")

# Calc Lightness cost-matrix
start = time.time()
l_centers = np.unique(center_clrs_real_lab[:,0])    # grab first val (L-val) from each center-color
light1 = l_centers[:, np.newaxis]
light2 = l_centers[np.newaxis, :]
cost_matrix_light = np.abs(light2 - light1)     # using None since .T does not work for 1D Arrays
print(f"Computed color cost-matrix in {time.time() - start:.3f} sec.")

# Save matrices
np.save(os.path.join(cwd, 'ColorSimilarity2', 'CostMatrix_full.npy'), cost_matrix_color, allow_pickle=True)
np.save(os.path.join(cwd, "ColorSimilarity", "CostMatrix_light.npy"), cost_matrix_light, allow_pickle=True)
