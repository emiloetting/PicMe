from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA
import os
import numpy as np
import json

# Load color embeddings
cwd = os.getcwd()
l2_color_path = os.path.join(cwd, "ColorSimilarity", "FullHists_L2.npy")
color_embeddings_l2 = np.load(l2_color_path)

# COLOR VISUALIZATION
#   2D:
#       PCA:
pca_2d = PCA(n_components=2)
color_embeddings_l2_pca_2d = pca_2d.fit_transform(color_embeddings_l2)

k_pca_2d = KernelPCA(n_components=2, kernel='rbf', eigen_solver='randomized')
color_embeddings_l2_k_pca_2d = k_pca_2d.fit_transform(color_embeddings_l2)

tsne_2d = TSNE(n_components=2, perplexity=30, max_iter=1000)
color_embeddings_l2_tsne_2d = tsne_2d.fit_transform(color_embeddings_l2)


# Store
dim_reduced_l2_color_embeds = {
    "pca_2d": color_embeddings_l2_pca_2d.tolist(),
    "k_pca_2d": color_embeddings_l2_k_pca_2d.tolist(),
    "tsne_2d": color_embeddings_l2_tsne_2d.tolist(),
}

json_path = os.path.join(cwd, "Analysis", "dim_reduced_l2_color_embeds.json")
with open(json_path, "w") as f:
    json.dump(dim_reduced_l2_color_embeds, f)