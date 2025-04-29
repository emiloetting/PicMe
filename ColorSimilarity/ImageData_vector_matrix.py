import os
import tqdm
from annoy import AnnoyIndex
import numpy as np
from colorClusterquantized import *



cwd = os.getcwd()
vector_matrix_path = os.path.join(cwd, 'ColorSimilarity', 'ImageData_vector_matrix.npy')

# Create quantized LAB colors
quantized_cosine_LAB, amount_colors = get_quantized_LAB(l_bins=8, a_bins=13, b_bins=13) # 8x13x13 = 1352 colors


# Use Annoy to create a nearest neighbor index for the quantized colors (instead of cKDTree)
annoy_index = AnnoyIndex(3, 'angular')  # 3 dimensions for LAB color space
for index, color in enumerate(quantized_cosine_LAB):
    annoy_index.add_item(index, quantized_cosine_LAB[index].astype(np.float32))    # Add quantized colors to the index
annoy_index.build(10)  # Build 10 trees for the index


# Setup paths
image_data_root = os.path.join(cwd, 'ImageData')
image_paths = [os.path.join(image_data_root, f) for f in os.listdir(image_data_root) if os.path.isfile(os.path.join(image_data_root, f))]

# Calculate Histograms
hists = []
with tqdm.tqdm(total=len(image_paths)) as bar:
    for file_path in image_paths:
        hist = quantized_image(file_path, annoy_index, quantized_cosine_LAB, 'L2')
        hists.append(hist)
        bar.update(1)


M_V = np.vstack(hists)  # Stack histograms vertically 

# Safe Vectors
np.save(vector_matrix_path, M_V, allow_pickle=True)
