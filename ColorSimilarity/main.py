import numpy as np
import os
from colorClusterquantized import *
from annoy import AnnoyIndex
import cv2 as cv
import matplotlib.pyplot as plt
import time




cwd = os.getcwd()
test_img_dir = os.path.join(cwd, 'ColorSimilarity', 'test_images')
image_data_root = os.path.join(cwd, 'ImageData')


# Setup path for image to find most similar images to
image_paths = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, img))]


# Setup quantized color spaces
quantized_cosine_LAB, amount_colors = get_quantized_LAB(l_bins=8, a_bins=13, b_bins=13) # for cosing similarity
quantized_EMD_LAB, amount_colors = get_quantized_LAB(l_bins=5, a_bins=5, b_bins=5) # for EMD


# Ignore (for visualization, calculated for displaying bar in color they are representing)
_lab = quantized_EMD_LAB.astype(np.uint8)[None, ...]              
_bgr = cv.cvtColor(_lab, cv.COLOR_Lab2BGR)[0]                     
rgb_EMD_colors = _bgr[..., ::-1].astype(np.float32) / 255.0


# Setup array of filepaths to find images later (will be found via index of this Array, as vectors were built in same order using same list)
database_image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]


# Prepare Annoy for Quantizing image to Cosine-LAB
annoy_index_cosine = AnnoyIndex(3, 'angular')  # 3 dimensions for LAB color space
for index, color in enumerate(quantized_cosine_LAB):
    annoy_index_cosine.add_item(index, quantized_cosine_LAB[index].astype(np.float32))    # Add quantized colors to the index
annoy_index_cosine.build(10)  # Build 10 trees for the index


# Prepare Annoy for Quantizing image to EMD-LAB
annoy_index_EMD = AnnoyIndex(3, 'angular')  
for index, color in enumerate(quantized_EMD_LAB):
    annoy_index_EMD.add_item(index, quantized_EMD_LAB[index].astype(np.float32))    # Add quantized colors to the index
annoy_index_EMD.build(10)  # Build 10 trees for the index


# Load cosine similarity matrix
M_V = np.load('ColorSimilarity/ImageData_vector_matrix.npy')


# Iterate through each image and find the most similar images
for i, img_path in enumerate(image_paths):

    # To quickly measure
    t_start = time.time()

    # Histogram Vector for image
    hist_vector = hist = quantized_image(img_path, annoy_index_cosine, quantized_cosine_LAB, 'L2') # Use L2 for finding cosine similarity

    # Calculate Cosine-Similarity
    distances = np.dot(M_V, hist_vector)  # Calculate cosine similarity between the histogram vector and all other histogram vectors in Database-Vectors

    # Find maximum cosine similarites
    idx_sorted = np.argsort(distances)[::-1]        # descending, so that the most similar image is first
    top5 = idx_sorted[idx_sorted != i][:5]          # Get the top 5 most similar images, excluding the input image itself
    min_index = top5
    
    t_cosine = time.time() - t_start
    print('Took ', t_cosine, ' seconds to calculate cosine distances')

    t_emd_start = time.time()

    # Collect file paths of the most similar images (according to cosine similarity)
    similar_images = [database_image_paths[i] for i in min_index]

    # Get EMD-Colorspace vector of input image
    input_EMD = quantized_image(img_path, annoy_index_EMD, quantized_EMD_LAB, 'L1')

    # Combine EMD histogram with indices for EMD calculation for input image
    weighted_bins_input = np.column_stack((
        input_EMD.astype(np.float32),
        np.arange(input_EMD.size, dtype=np.float32)
    ))

    hist_EMD_sims = []
    EMD_distances = []

    for sim_path in similar_images:
        hist = quantized_image(sim_path, annoy_index_EMD, quantized_EMD_LAB, 'L1')
        hist_EMD_sims.append(hist)

        # Copute EMD histogram for similar image
        weighted_bins_sim = np.column_stack((
            hist.astype(np.float32),
            np.arange(hist.size, dtype=np.float32)
        ))
        # Calculate EMD/Wasserstein
        dist, _, _ = cv.EMD(weighted_bins_input, weighted_bins_sim, cv.DIST_L2)
        EMD_distances.append(float(dist))

    # Sort by EMD distance (ascending -> most similar first)
    sim_list = list(zip(similar_images, distances[min_index], EMD_distances, hist_EMD_sims))
    sim_list.sort(key=lambda x: x[2])

    # Stop here (rest of the code is visualization)
    emd_time = time.time() - t_emd_start
    print('Took ', emd_time, ' seconds to calculate EMDs')


    # Visualization 
    # Create figure
    fig, axs = plt.subplots(2, 6, figsize=(18, 8))

    # Display input image
    axs[0, 0].imshow(cv.imread(img_path)[..., ::-1])
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Input')

    # Show it's EMD histogram (color displayed in rgb)
    in_hist = input_EMD
    for bi, idx in enumerate(np.argsort(in_hist)[::-1]):
        wt = in_hist[idx]
        axs[1, 0].bar(bi, wt, color=rgb_EMD_colors[idx], width=0.8)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_ylim(0, in_hist.max() * 1.2)

    # Do same but for top 5 candidates chosen via cosine similarity
    for col, (sim_path, cos_v, emd_v, hist_emd) in enumerate(sim_list, start=1):
        axs[0, col].imshow(cv.imread(sim_path)[..., ::-1])
        axs[0, col].set_title(f'#{col}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}')
        axs[0, col].axis('off')
    
        sorted_idx = np.argsort(hist_emd)[::-1]
        for bi, idx in enumerate(sorted_idx):
            wt = hist_emd[idx]
            axs[1, col].bar(bi, wt, color=rgb_EMD_colors[idx], width=0.8)
        axs[1, col].set_xticks([])
        axs[1, col].set_ylim(0, hist_emd.max() * 1.2)

    plt.tight_layout()
    plt.show()






