import os
import faiss
import json
import sqlite3
import pickle
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from annoy import AnnoyIndex
import tqdm as tqdm



if __name__ == "__main__":

    # Load embeds and cluster
    embeds = np.load("Analysis/clip_embeds.npy").astype(np.float32)
    kmeans = faiss.Kmeans(d=embeds.shape[1], 
                        k=25, 
                        niter=20, 
                        verbose=True, 
                        gpu=True, 
                        seed=92)
    kmeans.train(embeds)

    # get labels and centroids
    dist, labels = kmeans.index.search(embeds, k=1)
    centroids = kmeans.centroids

    # Load Annoy index to find NN of cluster centers
    ann_idx = AnnoyIndex(512, 'angular')
    ann_idx.load(os.path.join("DataBase", "clip_embeddings.ann"))

    # Find nearest neighbors for each cluster centroid
    centroid_neighbors = []
    for i in range(centroids.shape[0]):
        nn = ann_idx.get_nns_by_vector(centroids[i], 1)  # Get NN
        centroid_neighbors.append(nn[0])

    # Load json file to find respective img paths of output indices of centroid neighbors
    with open("DataBase/clip_embeddings_paths.json", "r") as f:
        img_paths = json.load(f)

    # Collect paths from json
    centroid_img_paths = []
    for nn in centroid_neighbors:
        centroid_img_paths.append(img_paths[str(nn)])

    # Collect 32*32 versions from hash_database.db 
    reduced_imgs = []
    conn = sqlite3.connect("DataBase/hash_database.db")
    cursor = conn.cursor()
    with tqdm.tqdm(total=len(centroid_img_paths)) as pbar:
        for img_path in centroid_img_paths:
            cursor.execute("SELECT * FROM image_hashes WHERE image_path=?", (img_path,))
            result = cursor.fetchone()
            if result is not None:
                reduced_imgs.append(pickle.loads(result[3]))  # Access image in third column
            else:
                print(f'Image not found in database: {img_path}')
            pbar.update(1)
    conn.close()

    # Rebuild imgs
    reduced_imgs = np.asarray(reduced_imgs) # in bgr
    reduced_imgs = reduced_imgs[..., ::-1]  # Convert to rgb

    # Dim-reduce centroids for plotting
    with open("Analysis/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    pca_centroids = pca.transform(centroids)

    print('Performing UMAP...')
    with open("Analysis/umap_model.pkl", "rb") as f:
        reducer = pickle.load(f)
    centroids_2d = reducer.transform(pca_centroids)

    # safe labels for plotting
    np.save("Analysis/cluster_labels.npy", labels.ravel())  # flatten for saving
    np.save("Analysis/cluster_centroids.npy", centroids_2d)
    np.save("Analysis/reduced_images_rgb.npy", reduced_imgs)