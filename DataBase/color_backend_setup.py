import os
import sys
import cv2
import time
import sqlite3
import annoy as ann
import numpy as np
from tqdm import tqdm
from skimage.color import deltaE_ciede2000
sys.path.append(os.path.join(os.getcwd(), 'ColorSimilarity'))
from colorClusterquantized import *


L_BINS = 5
A_BINS = 9
B_BINS = 9


if __name__ == "__main__":
    print("BUILDING COLOR-RELATED BACKEND")
    cwd = os.getcwd()


    # Setup paths
    component_subdir = os.path.join(cwd, 'DataBase', 'color_db_components')
    os.makedirs(component_subdir, exist_ok=True)

    image_data_root = os.path.join(cwd, 'ImageData')
    image_paths = [os.path.join(image_data_root, f) 
                for f in os.listdir(image_data_root) 
                if os.path.isfile(os.path.join(image_data_root, f))]


    # Collect hists
    full_hists_L1 = []
    full_hists_L2 = []
    image_sizes = []


    # Get hists
    with tqdm(total=len(image_paths)) as bar:
        for file_path in image_paths:
            img = cv2.imread(file_path)
            # Calc hists
            hist_full_complete_L1 = quantized_image(img, 
                                                    l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, 
                                                    normalization='L1', 
                                                    adjusted_bin_size=True)
            hist_full_complete_L2 = quantized_image(img, 
                                                    l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, 
                                                    normalization='L2')

            full_hists_L1.append(hist_full_complete_L1)
            full_hists_L2.append(hist_full_complete_L2)
            image_size = os.path.getsize(file_path)
            image_sizes.append(image_size)
            bar.update(1)


    # Stack 'em
    image_paths = np.asarray(image_paths)
    l1_embeds = np.stack(full_hists_L1)
    l2_embeds = np.stack(full_hists_L2)


    # Save hists
    np.save(os.path.join(cwd, 'DataBase', 'color_db_components', 'image_paths.npy'), 
            image_paths, allow_pickle=True)
    np.save(os.path.join(cwd, 'DataBase', 'color_db_components', 'FullHists_L1.npy'), 
            l1_embeds, allow_pickle=True)
    np.save(os.path.join(cwd, 'DataBase', 'color_db_components', 'FullHists_L2.npy'), 
            l2_embeds, allow_pickle=True)
    np.save(os.path.join(cwd, 'DataBase', 'color_db_components', 'image_sizes.npy'), 
            image_sizes, allow_pickle=True)


    dst_dir = os.path.join(cwd, 'DataBase', 'color_database.db')
    ann_dst_path = os.path.join(cwd, 'DataBase', 'color_ann_index.ann')

    # Remove existing database and index files to force rebuild
    if os.path.exists(dst_dir):
        os.remove(dst_dir)
        print("Removed existing database")

    if os.path.exists(ann_dst_path):
        os.remove(ann_dst_path)
        print("Removed existing Annoy index")

    # Create new database
    conn = sqlite3.connect(dst_dir)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE color_db (
            ann_index INTEGER PRIMARY KEY,
            path TEXT,
            L1_embedding BLOB,
            image_size INTEGER DEFAULT 0
            )
        ''')
    conn.commit()
    print("Created new database")




    # --------------CREATE COST MATRICES FOR EMD-------------------------------------------------------------------------------

    # Get bins in CIE-range
    center_clrs = get_bin_centers(l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS)
    center_clrs = center_clrs.reshape(-1, 1, 3) # reshape

    # Make explicit col-vector and row-vector to allow for vectorization 
    lab1 = center_clrs[:, np.newaxis, :]
    lab2 = center_clrs[np.newaxis, :, :]

    # Calc full cost-matrix
    print("Now calculating cost-matrix using Delta E 2000 metric")
    start = time.time()
    cost_matrix = deltaE_ciede2000(lab1, lab2).astype(np.float64)
    cost_matrix = np.squeeze(cost_matrix)   # get from shape (n,n,1) to (n,n) -> required format for EMD-calc
    end = time.time()
    print(f"Computed full cost-matrix in {end-start:.3f} sec.")

    # Save matrix
    np.save(os.path.join(cwd, 'DataBase', 'emd_cost_full.npy'), 
            cost_matrix, allow_pickle=True)



    # --------------Create ANNOY-INDEX and fill DB ----------------------------------------------------------------------------

    assert len(l1_embeds) == len(l2_embeds) == len(image_paths) == len(image_sizes), "All input arrays must have the same length"

    trees = 40
    dim = l2_embeds[0].shape[0]

    # [idx, path, l1, size]
    tuple_collection = []       # used to fill db later

    color_index = ann.AnnoyIndex(dim, 'angular')    # angular for euclidean -> cosine with L2 normalized

    with tqdm(total=len(l2_embeds), desc="Building Tuples, Filling Tree", unit="item") as pbar:
        for i in range(len(l2_embeds)):
            color_index.add_item(i, l2_embeds[i])   # L2 in ANNOY-Index
            img_tuple = (int(i), str(image_paths[i]), l1_embeds[i].astype(np.float64).tobytes(), int(image_sizes[i]))     # L1 in DB as float64
            tuple_collection.append(img_tuple)

            pbar.update(1)

    color_index.build(trees)
    color_index.save(ann_dst_path)

    c.executemany(
        "INSERT INTO color_db (ann_index, path, L1_embedding, image_size) VALUES (?, ?, ?, ?)",
        tuple_collection
    )
    conn.commit()
    conn.close()
