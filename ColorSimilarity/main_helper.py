import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cv2 as cv
from ColorSimilarity.colorClusterquantized import *
import time
from pyemd import emd_with_flow
import annoy as ann
import os
import sqlite3

# Try to import from DataBase package, fall back to hardcoded values
try:
    from DataBase.color_backend_setup import L_BINS, A_BINS, B_BINS
except ModuleNotFoundError:
    from backend_setup import L_BINS, A_BINS, B_BINS



ann_idx_path = os.path.join(os.getcwd(), 'DataBase', 'color_ann_index.ann')
ann_idx = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
ann_idx.load(ann_idx_path)
cwd = os.getcwd()


# HELPER-MODULE CONTAINING FUNCTIONS FOR MAIN.PY----------------------------------


def presort_sims(sims: NDArray, relative_tresh: float) -> list[NDArray]:
    """Drops vals in distances if vals are smaller than highest value time relative threshhold, sorts remaining values -> Computing more effectively"
    
    Args: 
        sims (NDArray): Array containing cosine-similarities of color embeddings and input image.
        relative_thresh (float): Factor used to decide whether or not to drop a value in sims. 
    
    Returns:
        idx_and_dist (list(NDArray)): List with 2 elements: indices and distances
    """
    assert sims.ndim == 1, f"Expected (n,), got {sims.shape}"

    # Find maximum similarities
    max_sim = np.max(sims)

    # Create mask
    mask = sims > (relative_tresh * max_sim)

    # Get indices and distances
    filtered_indices = np.where(mask)[0]
    filtered_distances = sims[mask]

    return [filtered_indices, filtered_distances]


def calc_emd_and_final(input_vec_l1: NDArray,
                        cosine_values: NDArray, 
                        l1_embeds: NDArray, 
                        emd_cost_mat: NDArray|None) -> list[list]:   
        
    """ Calculates the EMD and final metric for the given indices and similar images."""
    emd_values = []
    final_metrics = []

    for i, l1_hist in enumerate(l1_embeds):
        db_hist = np.frombuffer(l1_hist, dtype=np.float64)
        dist , _ = emd_with_flow(input_vec_l1, db_hist, distance_matrix=emd_cost_mat)
        emd_values.append(float(dist))
        final_metrics.append(dist / max(cosine_values[i], 1e-9))

    return [emd_values, final_metrics]

def work_that_vectors_ann(input_vec_l1: NDArray,
                    input_vec_l2: NDArray,
                    input_img_paths: list[str],
                    annoy_index: ann.AnnoyIndex, 
                    emd_cost_mat: NDArray|None, 
                    img_weights: list[float]|None=None,
                    num_results: int=12,  # Statt batch_size - Anzahl der Ergebnisse
                    emd_count: int=12,    # Anzahl der EMD-Berechnungen
                    track_time: bool=False,
                    show: bool=False) -> list[str]:
    """
    Processes the input image vectors and finds similar images using Annoy index.

    Args:
        input_vec_l1 (NDArray): The L1 feature vector of the input image.
        input_vec_l2 (NDArray): The L2 feature vector of the input image.
        input_img_paths (list[str]): The file paths of the input images.
        db_img_paths (list[str]): The file paths of the database images.
        l1_emb (NDArray): The L1 embeddings of the database images.
        annoy_index (ann.AnnoyIndex): The Annoy index for fast nearest neighbor search.
        emd_cost_mat (NDArray|None): The cost matrix for EMD calculation.
        img_weights (list[float]|None): Weights for each image's contribution to the final metric.
        num_results (int): Number of similar images to return (default: 12).
        emd_count (int): Number of images for which to calculate EMD (default: 12).
        track_time (bool): Whether to track processing time.
        show (bool): Whether to display results in matplotlib window.

    Returns:
        list[str]: Paths of images ordered by similarity.
    """
    
    if track_time:
        start = time.time()

    # Get top N images according to annoy
    indices, distances = annoy_index.get_nns_by_vector(input_vec_l2, num_results, include_distances=True)
    cosine_values = 1 - np.array(distances)**2 / 2  # make cosine sim from angular distances

    # Call to DB for L1 embeddings
    dst_dir = os.path.join(cwd, 'DataBase', 'color_database.db')
    connection = sqlite3.connect(dst_dir)
    cursor = connection.cursor()

    placeholders = ','.join('?' for _ in indices)
    query = f"SELECT L1_embedding, path FROM color_db WHERE ann_index IN ({placeholders})"
    cursor.execute(query, tuple(indices))
    results = cursor.fetchall()

    # split into desired lists
    l1_embeddings = [row[0] for row in results]
    sim_ordered_paths = [row[1] for row in results]


    if track_time:
        print(f"Annoy search time: {time.time()-start:.3f}s")
        start_emd = time.time()

    # Berechne EMD für die angegebene Anzahl von Bildern
    # Nutze min() um sicherzustellen, dass wir nicht mehr Bilder verarbeiten als wir haben
    emd_calc_size = min(emd_count, len(indices))
    top_cosine_values = cosine_values[:emd_calc_size]
    
    emd_values, final_metrics = calc_emd_and_final(
        input_vec_l1, 
        top_cosine_values, 
        l1_embeddings, 
        emd_cost_mat
    )
    

    # Sortiere die Bilder mit EMD-Berechnung nach final_metrics
    top_paths = [sim_ordered_paths[i] for i in range(emd_calc_size)]
    sorted_indices = np.argsort(final_metrics)  # sort ascending by final metrics
    sorted_top_paths = [top_paths[i] for i in sorted_indices]
    
    
    # Ersetze die entsprechenden Einträge in sim_ordered_paths mit den sortierten
    sim_ordered_paths[:emd_calc_size] = sorted_top_paths
    
    if track_time and emd_calc_size > 0:
        print(f"EMD time: {time.time()-start_emd:.3f}s\n")
    
    if show:
        visualize(
            input_img_paths, 
            sim_ordered_paths[:12], 
            cosine_values[:12],      
            emd_values,
            final_metrics,
            sort_by='final', 
            img_weights=img_weights
        )

    # Liste der Pfade in absteigender Ähnlichkeit
    return sim_ordered_paths  # Return only the top 12 results for visualization

    
def color_match_single_ann(img_path: list[str], 
                        annoy_index: ann.AnnoyIndex, 
                        l_bins: int, a_bins: int, b_bins: int, 
                        emd_cost_mat: NDArray|None, 
                        num_results: int=12,
                        emd_count: int=12,
                        track_time: bool=False,
                        show: bool=False,
                        adjusted_bin_size: bool=False) -> list[str]:

    """Finds the most similar image in the database for a single input image based on cosine similarity and EMD of LAB-space histograms.

    Args:
        img_path (list(str)): List containing the path to the input image. Must contain exactly one image path.
        db_img_paths (list[str]): List of paths to the database images.
        l_bins (int): Number of bins for the L channel.
        a_bins (int): Number of bins for the A channel.
        b_bins (int): Number of bins for the B channel.
        track_time (bool): Whether to track and print the time taken for each step.

    Returns:
        list[str]: Paths to the most similar images in order of similarity.
    """
    assert isinstance(img_path, list) and len(img_path) == 1, f"img_path must be a list containing exactly one image path, got {img_path}"

    # Keep track of time if desired
    if track_time:
        start = time.time()
    img = cv2.imread(img_path[0])
    input_vector_l2 = quantized_image(img, l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L2')
    input_vector_l1 = quantized_image(img, l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L1', adjusted_bin_size=adjusted_bin_size).astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims and return the result
    return work_that_vectors_ann(input_vec_l1=input_vector_l1, 
                      input_vec_l2=input_vector_l2, 
                      input_img_paths=img_path,
                      annoy_index=annoy_index, 
                      emd_cost_mat=emd_cost_mat,
                      num_results=num_results,  # Use batch_size as number of results
                      emd_count=emd_count,  # Use batch_size for EMD calculations
                      track_time=track_time,
                      show=show)
    

def color_match_double_ann(img_paths: list[str], 
                        annoy_index: ann.AnnoyIndex, 
                        l_bins: int, a_bins: int, b_bins: int, 
                        emd_cost_mat: NDArray|None, 
                        img_weights: list[float]=[1.0, 1.0],
                        num_results: int=12,
                        emd_count: int=12,
                        track_time: bool=False,
                        show: bool=False,
                        adjusted_bin_size: bool=False) -> NDArray:

    """Finds the most similar image in the database for a pair of input images based on cosine similarity and EMD of LAB-space histograms.

    Args:
        img_paths (list[str]): List containing the paths to the input images. Must contain exactly two image paths.
        db_img_paths (list[str]): List of paths to the database images.
        l1_emb (NDArray): L1 embedding matrix.
        annoy_index (AnnoyIndex): Annoy index for fast nearest neighbor search.
        l_bins (int): Number of bins for the L channel.
        a_bins (int): Number of bins for the A channel.
        b_bins (int): Number of bins for the B channel.
        emd_cost_mat (NDArray|None): Cost matrix for EMD calculation.
        img_weights (list[float]): Weights for each image's histogram contribution. Sum must be 2.
        batch_size (int, optional): Size of the batches for processing. Defaults to 10.
        track_time (bool, optional): Whether to track processing time and display in terminal. Defaults to False.
        show (bool, optional): Whether to show visualization in matplotlib. Defaults to False.

    Returns:
        NDArray: Array containing the similarity scores for each database image.
    """
    assert isinstance(img_paths, (list, tuple)) and len(img_paths) == 2, f"img_paths must be a list containing exactly two image paths, got {img_paths}"

    # Extract paths
    img1, img2 = cv2.imread(img_paths[0]), cv2.imread(img_paths[1])

    # Track time if desired
    if track_time:
        start = time.time()

    # Calc L1 and L2 normalized vectors for both images
    combined_vec_l1 = quantize2images(images=[img1, img2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L1', adjusted_bin_size=adjusted_bin_size, weights=img_weights).astype(np.float64)

    combined_vec_l2 = quantize2images(images=[img1, img2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L2', adjusted_bin_size=False, weights=img_weights).astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims
    return work_that_vectors_ann(input_vec_l1=combined_vec_l1, 
                      input_vec_l2=combined_vec_l2, 
                      input_img_paths=img_paths,
                      annoy_index=annoy_index, 
                      emd_cost_mat=emd_cost_mat,
                      num_results=num_results,  # Use batch_size as number of results
                      emd_count=emd_count,  # Use batch_size for EMD calculations
                      track_time=track_time,
                      show=show)