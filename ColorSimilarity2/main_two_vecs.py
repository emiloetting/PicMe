import numpy as np
import os
from colorClusterquantized import *
import cv2 as cv
import matplotlib.pyplot as plt
from pyemd import emd_with_flow



cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity2', 'test_images')
img_vec_mat_pos_path = os.path.join(cwd, 'ColorSimilarity2', 'PosHists.npy')
img_vec_mat_neg_path = os.path.join(cwd, 'ColorSimilarity2', 'NegHists.npy')
cost_mat_path = os.path.join(cwd, "ColorSimilarity2", "EMDcosts.npy")
image_data_root = os.path.join(cwd, 'ImageData')


# Setup path for image to find most similar images to
image_paths = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, img))]

# Setup array of filepaths to find images later (will be found via index of this Array, as vectors were built in same order using same list)
database_image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Load img-vector matrices
M_V_pos = np.load(img_vec_mat_pos_path)
M_V_neg = np.load(img_vec_mat_neg_path)

print(np.linalg.norm(M_V_pos[0] - M_V_pos[1]))
print(np.linalg.norm(M_V_neg[0] - M_V_neg[1]))

# Load cost-matrix for EMD
cost_matrix = cost = np.load(cost_mat_path)


# Iterate through each image and find the most similar images
for i, img_path in enumerate(image_paths):

    # Histogram Vector for image
    hist_pos, hist_neg = quantized_image_two_vecs(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2') # Use L2 for finding cosine similarity

    norm = np.linalg.norm(hist_neg, ord=2)
    if norm > 0:
        hist_neg /= norm
    else:
        hist_neg[:] = 0  # oder eine andere sinnvolle Default-Behandlung
    
    # Calc distances
    distances_pos = np.dot(M_V_pos,hist_pos)
    distances_neg = np.dot(M_V_neg, hist_neg)

    # Collect best suited images
    combined_score = distances_pos + distances_neg
    idx_sorted = np.argsort(combined_score)[::-1]
    min_index = idx_sorted[:5]

    # Collect file paths of the most similar images (according to cosine similarity)
    similar_images = [database_image_paths[i] for i in min_index]

    # Get cosine values for the top 5
    cosine_values = [combined_score[i] for i in min_index]

    # Get EMD-Colorspace vector of input image
    input_EMD = quantized_image(img_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1').astype(np.float64)

    hist_EMD_sims = []
    EMD_distances = []

    for sim_path in similar_images:
        hist = quantized_image(sim_path, l_bins=int(L_BINS), a_bins=int(A_BINS), b_bins=int(B_BINS), normalization='L1').astype(np.float64)
        hist_EMD_sims.append(hist)
        
        # Calculate EMD/Wasserstein
        dist, _ = emd_with_flow(input_EMD, hist, cost_matrix)
        EMD_distances.append(float(dist))

    # Sort by EMD distance (ascending -> most similar first)
    sim_list = list(zip(similar_images, cosine_values, EMD_distances, hist_EMD_sims))
    sim_list.sort(key=lambda x: x[2])  # Sort by EMD distance






    # Visualization 
    # Create figure
    fig, axs = plt.subplots(1, 6, figsize=(18, 8))

    # Display input image
    axs[0].imshow(cv.imread(img_path)[..., ::-1])
    axs[0].axis('off')
    axs[0].set_title('Input')


    # Top 5 Kandidaten (nach EMD sortiert)
    for col, (sim_path, cos_v, emd_v, hist_emd) in enumerate(sim_list, start=1):
        # Bild anzeigen
        axs[col].imshow(cv.imread(sim_path)[..., ::-1])
        axs[col].set_title(f'#{col}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}')
        axs[col].axis('off')
        

    plt.tight_layout()
    plt.show()






