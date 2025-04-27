import os
import tqdm
from annoy import AnnoyIndex
import numpy as np
from colorClusterquantized import *



cwd = os.getcwd()
sim_matrix_path = os.path.join(cwd, 'ColorSimilarity', 'cosine_similarity_matrix.npy')

# Create quantized LAB colors
quantized_cosine_LAB, amount_colors = get_quantized_LAB(l_bins=8, a_bins=13, b_bins=13) # 8x13x13 = 1352 colors


# Use Annoy to create a nearest neighbor index for the quantized colors (instead of cKDTree)
annoy_index = AnnoyIndex(3, 'angular')  # 3 dimensions for LAB color space
for index, color in enumerate(quantized_cosine_LAB):
    annoy_index.add_item(index, quantized_cosine_LAB[index].astype(np.float32))    # Add quantized colors to the index
annoy_index.build(10)  # Build 10 trees for the index


# Create Matrix of cosine distances between quantized image-vectors
image_data_root = os.path.join(cwd, 'ImageData')
image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Calculate Histograms
hists = []
with tqdm.tqdm(total=len(image_paths)) as bar:
    for file_path in image_paths:
        hist = quantized_image(file_path, annoy_index, quantized_cosine_LAB, 'L2')
        hists.append(hist)
        bar.update(1)


M_H = np.vstack(hists)  # Stack histograms vertically 
sim_matrix = M_H @ M_H.T  # Calculate cosine similarity matrix -> works, because the histograms are L2-normalized -> ||a|| = ||b|| = 1 and therefor each divisor is 1

# Safe matrix to file
np.save(sim_matrix_path, sim_matrix, allow_pickle=True)

print('Shape of the similarity matrix:', sim_matrix.shape)