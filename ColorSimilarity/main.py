import numpy as np
import os
from colorClusterquantized import *
import cv2 as cv
import matplotlib.pyplot as plt
from pyemd import emd_with_flow
import time



cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity', 'test_images')
test_img_dir = os.path.join(cwd, 'ColorSimilarity', 'DAISY_25')
full_hists_L1 = os.path.join(cwd, 'ColorSimilarity', 'FullHists_L1.npy')
full_hists_L2 = os.path.join(cwd, 'ColorSimilarity', 'FullHists_L2.npy')
cost_mat_full_path = os.path.join(cwd, "ColorSimilarity", 'CostMatrix_full.npy')
image_data_root = os.path.join(cwd, 'ImageData')


# Setup path for image to find most similar images to
image_paths = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, img))]

# Setup array of filepaths to find images later (will be found via index of this Array, as vectors were built in same order using same list)
database_image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Load img-vector matrix
M_V_L1 = np.load(full_hists_L1)
M_V_L2 = np.load(full_hists_L2)

# Load cost-matrix for EMD
cost_matrix_full = np.load(cost_mat_full_path)


# Iterate through each image and find the most similar images
for i, img_path in enumerate(image_paths):

    # Start timer for time-inspection in terminal
    start = time.time()

    # Histogram Vector for image
    hist_vector = quantized_image(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
    print(f'Hist for input took: {time.time()-start:.3f}s')
    distances = np.dot(M_V_L2,hist_vector)

    # Pre-Sort & Find maximum cosine similarites
    max_sim = np.max(distances)
    mask = distances > (0.5 * max_sim)
    filtered_indices = np.where(mask)[0]
    filtered_distances = distances[mask]
    
    # Sort remainaing imgs
    idx_sorted = np.argsort(filtered_distances)[::-1]       # descending, so that the most similar image is first
    min_index = filtered_indices[idx_sorted[:10]]            # Get the top 5 most similar images

    # Collect file paths of the most similar images (according to cosine similarity)
    similar_images = [database_image_paths[i] for i in min_index]

    # Get cosine values for the top 5
    cosine_values = filtered_distances[idx_sorted[:10]]
    print(f"Cosine time:{time.time()-start:.3f}s")  # Display required time 

    # Time EMD-Calculations
    start_emd = time.time()

    # Init empty list to keep distances to top 10 selected imgs according to cosine-sim
    EMD_full_distances = []

    # List for weighted full decisive metric
    final_metric = []

    # Input L1
    hist_vector_L1 = quantized_image(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1').astype(np.float64)

    # For each image in Top 10
    for i, (idx, sim_path) in enumerate(zip(min_index, similar_images)):
        
        # Get complete EMD for 
        db_full_hist = M_V_L1[idx].astype(np.float64)

        dist , _ = emd_with_flow(hist_vector_L1, db_full_hist, cost_matrix_full)
        EMD_full_distances.append(float(dist))

        final_metric.append(dist / max(cosine_values[i], 1e-9))

    # Sort by EMD distance (ascending -> most similar first)
    # cosine_values_updated = distances_full_vecs  # passt zu dem, was du sortierst
    sim_list = list(zip(similar_images, cosine_values, EMD_full_distances, final_metric))
    sim_list.sort(key=lambda x: x[3])  # Sort by EMD distance

    # Display duration of calculatin time
    print(f"EMDs: {time.time()-start_emd:.3f}s\n")



    # Visualization 
    # Create figure
    fig, axs = plt.subplots(1, 6, figsize=(18, 8))

    # Display input image
    axs[0].imshow(cv.imread(img_path)[..., ::-1])
    axs[0].axis('off')
    axs[0].set_title('Input')

    # Top 5 Kandidaten (nach EMD sortiert)
    for col, (sim_path, cos_v, emd_v, total_score) in enumerate(sim_list[:5], start=1):
        axs[col].imshow(cv.imread(sim_path)[..., ::-1])
        axs[col].set_title(f'#{col}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}\nTotal Score: {total_score:.3f}')
        axs[col].axis('off')
        
    plt.tight_layout()
    plt.show()






