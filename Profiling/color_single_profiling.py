import os
import sys
import numpy as np
from ColorSimilarity.main_helper import *
from ColorSimilarity.colorClusterquantized import L_BINS, A_BINS, B_BINS


if __name__ == "__main__":
    cwd = os.getcwd()
    full_hists_L1 = os.path.join(cwd, 'ColorSimilarity', 'FullHists_L1.npy')
    image_path = os.path.join(cwd,'Profiling', 'test_img_1.jpg')
    cost_mat_path = os.path.join(cwd, "ColorSimilarity", 'CostMatrix_full.npy')
    ann_idx_path = os.path.join(cwd, 'ColorSimilarity', 'color_index_l2.ann')
    image_data_root = os.path.join(cwd, 'ImageData')
    database_image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]  

    # Load cost matrix for EMD & L1 normalized backend vecs
    cost_matrix = np.load(cost_mat_path)
    M_V_L1 = np.load(full_hists_L1)

    # Load ANN tree
    l2_index = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
    l2_index.load(ann_idx_path)

    # Perform actual similarity search
    color_match_single_ann(
        img_path=[image_path],
        db_img_paths=database_image_paths,
        l1_emb=M_V_L1,
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
    sys.exit()