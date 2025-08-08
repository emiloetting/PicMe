import os
import tqdm
import numpy as np
from colorClusterquantized import *
from skimage.color import deltaE_ciede2000
import time


cwd = os.getcwd()


# BUILD HISTOGRAMS

# Setup paths
image_data_root = os.path.join(cwd, 'ImageData')
image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Collect hists
# ab_hists = []
# lightness_hists = []
# full_hists_signed = []
full_hists_L1 = []
full_hists_L2 = []

# Get hists
with tqdm.tqdm(total=len(image_paths)) as bar:
    for file_path in image_paths:

        # Calc hists
        # hist_ab, hist_l = quantized_image_two_vecs(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        # hist_full_signed = quantized_image_signed(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        hist_full_complete_L1 = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1')
        hist_full_complete_L2 = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
        # Sort hists
        # ab_hists.append(hist_ab)
        # lightness_hists.append(hist_l)
        # full_hists_signed.append(hist_full_signed)
        full_hists_L1.append(hist_full_complete_L1)
        full_hists_L2.append(hist_full_complete_L2)
        bar.update(1)

# Stack 'em
# M_V_ab = np.vstack(ab_hists)
# M_V_light = np.vstack(lightness_hists)
# M_V_signed = np.stack(full_hists_signed)
M_V_L1 = np.stack(full_hists_L1)
M_V_L2 = np.stack(full_hists_L2)

# Save hists
# np.save(os.path.join(cwd, 'ColorSimilarity3', 'ABHists.npy'), M_V_ab, allow_pickle=True)
# np.save(os.path.join(cwd, 'ColorSimilarity3', 'LightHists.npy'), M_V_light, allow_pickle=True)
# np.save(os.path.join(cwd, 'ColorSimilarity3', 'FullHists_Signed.npy'), M_V_signed, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity3', 'FullHists_L1.npy'), M_V_L1, allow_pickle=True)
np.save(os.path.join(cwd, 'ColorSimilarity3', 'FullHists_L2.npy'), M_V_L2, allow_pickle=True)
#--------------------------------------------------------------------------------------------------------------------------

# CREATE COST MATRICES FOR EMD

# Get bins in CIE-range
center_clrs = get_bin_centers(l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS)
center_clrs = center_clrs.reshape(-1, 1, 3) # reshape

# Make explicit col-vector and row-vector to allow for vectorization 
lab1 = center_clrs[:, np.newaxis, :]
lab2 = center_clrs[np.newaxis, :, :]

# Calc full cost-matrix
print("Now calculating cost-matrix using Delta E 2000 metric")
start = time.time()
cost_matrix = deltaE_ciede2000(lab1, lab2).astype(np.float64)
cost_matrix = np.squeeze(cost_matrix)   # get from shape (n,n,1) to (n,n) -> 

end = time.time()
print(f"Computed full cost-matrix in {end-start:.3f} sec.")

# Save matrix
np.save(os.path.join(cwd, 'ColorSimilarity3', 'CostMatrix_full.npy'), cost_matrix, allow_pickle=True)