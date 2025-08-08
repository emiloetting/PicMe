import numpy as np
import os
from colorClusterquantized import *
import cv2 as cv
import matplotlib.pyplot as plt
from pyemd import emd_with_flow
import time



cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity3', 'test_images')
full_hists_L1 = os.path.join(cwd, 'ColorSimilarity3', 'FullHists_L1.npy')
full_hists_L2 = os.path.join(cwd, 'ColorSimilarity3', 'FullHists_L2.npy')
cost_mat_full_path = os.path.join(cwd, "ColorSimilarity3", 'CostMatrix_full.npy')
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
    distances = np.dot(M_V_L2,hist_vector)

    # Pre-Sort & Find maximum cosine similarites
    max_sim = np.max(distances)
    mask = distances > (0.3 * max_sim)
    filtered_indices = np.where(mask)[0]
    filtered_distances = distances[mask]

    # Calc full vec for all vecs mit cos-sim of more than half of best cos-sim found
    full_input_vec = quantized_image(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')
    full_input_vec = full_input_vec.astype(np.float64)
    distances_full_vecs = np.dot(M_V_L2[filtered_indices], full_input_vec)
    
    # Sort remainaing imgs
    idx_sorted = np.argsort(distances_full_vecs)[::-1]       # descending, so that the most similar image is first
    min_index = filtered_indices[idx_sorted[:10]]            # Get the top 5 most similar images

    # Collect file paths of the most similar images (according to cosine similarity)
    similar_images = [database_image_paths[i] for i in min_index]

    # Get cosine values for the top 5
    cosine_values = [distances[i] for i in min_index]
    print(f"Cosine time:{time.time()-start:.3f}s")  # Display required time 

    # Time EMD-Calculations
    start = time.time()

    # # Calculate hist for input image
    # input_hist_ab, input_hist_l = quantized_image_two_vecs(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1')
    # input_hist_ab = input_hist_ab.astype(np.float64)
    # input_hist_l = input_hist_l.astype(np.float64)

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

        final_metric.append(dist / (1*cosine_values[i]))

    # Sort by EMD distance (ascending -> most similar first)
    cosine_values_updated = distances_full_vecs  # passt zu dem, was du sortierst
    sim_list = list(zip(similar_images, cosine_values_updated, EMD_full_distances, final_metric))
    sim_list.sort(key=lambda x: x[3])  # Sort by EMD distance

    # Display duration of calculatin time
    print(f"EMDs: {time.time()-start:.3f}s\n")



    # Visualization 
    # Create figure
    fig, axs = plt.subplots(1, 6, figsize=(18, 8))

    # Display input image
    axs[0].imshow(cv.imread(img_path)[..., ::-1])
    axs[0].axis('off')
    axs[0].set_title('Input')

    # Top 5 Kandidaten (nach EMD sortiert)
    for col, (sim_path, cos_v, emd_v, total_score) in enumerate(sim_list[:5], start=1):

        # Bild anzeigen
        axs[col].imshow(cv.imread(sim_path)[..., ::-1])
        axs[col].set_title(f'#{col}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}\nTotal Score: {total_score:.3f}')
        axs[col].axis('off')
        
    plt.tight_layout()
    plt.show()






