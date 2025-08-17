import os
import numpy as np
from ColorSimilarity.main_helper import *
from Initialization.setup import L_BINS, A_BINS, B_BINS


if __name__ == "__main__":
    cwd = os.getcwd()
    image_path = os.path.join(cwd,'Profiling', 'test_img_1.jpg')
    cost_mat_path = os.path.join(cwd, "DataBase", 'emd_cost_full.npy')
    ann_idx_path = os.path.join(cwd, 'DataBase', 'color_ann_index.ann')  

    # Load cost matrix for EMD 
    cost_matrix = np.load(cost_mat_path)

    # Load ANN tree
    l2_index = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
    l2_index.load(ann_idx_path)

    # Perform actual similarity search
    color_match_single_ann(
        img_path=[image_path],
        annoy_index=l2_index,
        l_bins=L_BINS,
        a_bins=A_BINS,
        b_bins=B_BINS,
        emd_cost_mat=cost_matrix,
        num_results=12,  # Use 12 as number of results
        emd_count=12,  # Use 12 for EMD calculations
        track_time=True,
        show=False,
        adjusted_bin_size=True
    )