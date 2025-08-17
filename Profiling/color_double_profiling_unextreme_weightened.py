import os
import numpy as np
from ColorSimilarity.main_helper import *
from Initialization.setup import L_BINS, A_BINS, B_BINS


if __name__ == "__main__":
    cwd = os.getcwd()
    image_path_1 = os.path.join(cwd,'Profiling', 'test_img_1.jpg')
    image_path_2 = os.path.join(cwd,'Profiling', 'test_img_2.jpg')
    cost_mat_path = os.path.join(cwd, "DataBase", 'emd_cost_full.npy')
    ann_idx_path = os.path.join(cwd, 'DataBase', 'color_ann_index.ann')   

    # Load cost matrix for EMD
    cost_matrix = np.load(cost_mat_path)

    # Load ANN tree
    l2_index = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
    l2_index.load(ann_idx_path)

    color_match_double_ann(
        img_paths=[image_path_1, image_path_2],
        annoy_index=l2_index,
        l_bins=L_BINS,
        a_bins=A_BINS,
        b_bins=B_BINS,
        emd_cost_mat=cost_matrix,
        img_weights=[.5, 1.5],
        num_results=12,
        emd_count=12,
        track_time=True,
        show=False,
        adjusted_bin_size=True
    )