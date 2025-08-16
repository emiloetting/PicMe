from sklearn.decomposition import PCA
import os
import numpy as np
from annoy import AnnoyIndex
import numpy as np
from umap import UMAP
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    cwd = os.getcwd()


    # Grab CLIP embeddings
    clip_embed_dims = 512
    ann_idx = AnnoyIndex(clip_embed_dims, 'angular')
    ann_idx.load(os.path.join("DataBase", "clip_embeddings.ann"))
    amnt_files = ann_idx.get_n_items()
    clip_vectors = np.zeros((amnt_files, clip_embed_dims), dtype=np.float32)
    with tqdm(total=amnt_files, desc="Extracting + normalizing CLIP embeddings") as pbar:
        for i in range(amnt_files):
            vec = np.array(ann_idx.get_item_vector(i), dtype=np.float32)
            clip_vectors[i] = vec / np.linalg.norm(vec)
            pbar.update(1)

    # Save embeddings
    np.save(os.path.join(cwd, "Analysis", "clip_embeds.npy"), clip_vectors)

    # First step: Rough reduction using PCA
    pca = PCA(n_components=30)
    clip_pca = pca.fit_transform(clip_vectors)

    # Second step: more granular using UMAP
    print('Performing UMAP...')
    reducer = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
        verbose=True,
        metric='cosine'
    )
    clip_umap = reducer.fit_transform(clip_pca)

    # Save final reduced embeddings
    final_path = os.path.join(cwd, "Analysis", "dim_reduced_clip_embeds.npy")
    np.save(final_path, clip_umap)

    # Safe PCA and UMAP model for applying later to centroids
    with open(os.path.join(cwd, "Analysis", "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(cwd, "Analysis", "umap_model.pkl"), "wb") as f:
        pickle.dump(reducer, f)