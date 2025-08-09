import numpy as np
import os
from colorClusterquantized import *
from main_helper import *


# Declare amount of bins
L_BINS = 5
A_BINS = 15
B_BINS = 15


# Setup paths for image data
cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity', 'test_images')
full_hists_L1 = os.path.join(cwd, 'ColorSimilarity', 'FullHists_L1.npy')
full_hists_L2 = os.path.join(cwd, 'ColorSimilarity', 'FullHists_L2.npy')
cost_mat_path = os.path.join(cwd, "ColorSimilarity", 'CostMatrix_full.npy')
image_data_root = os.path.join(cwd, 'ImageData')

# Setup path for image to find most similar images to
image_paths = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, img))]
image_paths_reversed = list(reversed(image_paths))[80:]

# Setup array of filepaths to find images later (will be found via index of this Array, as vectors were built in same order using same list)
database_image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Load img-vector matrix
M_V_L1 = np.load(full_hists_L1)
M_V_L2 = np.load(full_hists_L2)

# Load cost-matrix for EMD
cost_matrix = np.load(cost_mat_path)

single_double = 'double'

# Iterate through each image and find the most similar images
if single_double == 'single':
    for img_path in image_paths:
        color_match_single(
            img_path=[img_path],
            db_img_paths=database_image_paths,
            l1_emb=M_V_L1,
            l2_emb=M_V_L2,
            l_bins=L_BINS,
            a_bins=A_BINS,
            b_bins=B_BINS,
            emd_cost_mat=cost_matrix,
            batch_size=10,
            track_time=True,
            show=True
        )

if single_double == 'double':
    for img_path in zip(image_paths, image_paths_reversed):
        color_match_double(
            img_paths=img_path,
            db_img_paths=database_image_paths,
            l1_emb=M_V_L1,
            l2_emb=M_V_L2,
            l_bins=L_BINS,
            a_bins=A_BINS,
            b_bins=B_BINS,
            emd_cost_mat=cost_matrix,
            batch_size=10,
            track_time=True,
            show=True
        )
