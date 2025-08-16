import numpy as np
import os
from ColorSimilarity.colorClusterquantized import *
from ColorSimilarity.main_helper import *
from DataBase.color_backend_setup import L_BINS, A_BINS, B_BINS




# Setup paths for image data
cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity', 'test_images')
test_img_paths = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, img))]

# Load up ANN index
ann_idx_path = os.path.join(cwd, 'DataBase', 'color_ann_index.ann')
l2_index = ann.AnnoyIndex(L_BINS * A_BINS * B_BINS, 'angular')
l2_index.load(ann_idx_path)


# Load cost-matrix for EMD
cost_mat_path = os.path.join(cwd, 'DataBase', 'emd_cost_full.npy')
cost_matrix = np.load(cost_mat_path)

single_double = 'double'

# Iterate through each image and find the most similar images
if single_double == 'single':
    for img_path in test_img_paths:
        color_match_single_ann(
            img_path=[img_path],
            annoy_index=l2_index,
            l_bins=L_BINS,
            a_bins=A_BINS,
            b_bins=B_BINS,
            emd_cost_mat=cost_matrix,
            num_results=12,  # Use 12 as number of results
            emd_count=12,  # Use 12 for EMD calculations
            track_time=True,
            show=True,
            adjusted_bin_size=True
        )

if single_double == 'double':
    for img_path in zip(test_img_paths, test_img_paths[::-1]):
        color_match_double_ann(
            img_paths=img_path,
            annoy_index=l2_index,
            l_bins=L_BINS,
            a_bins=A_BINS,
            b_bins=B_BINS,
            emd_cost_mat=cost_matrix,
            img_weights=[.5, 1.5],
            num_results=12,
            emd_count=12,
            track_time=True,
            show=True,
            adjusted_bin_size=True
        )
        
