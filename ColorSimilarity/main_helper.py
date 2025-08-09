import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cv2 as cv
from colorClusterquantized import *
import time
from pyemd import emd_with_flow



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


def visualize(img_path: list[str], sim_imgs: list, cos_vals: NDArray, emd_dists: list[float]|None, final_metric: NDArray|None, sort_by: str):
    """Opens matplotlib visualization of top-5 images based on cosine similarities and EMD histograms.

    Args:
        img_path (str|list[str]): Path to the input image or list of image paths.
        sim_imgs (list): List of similar images.
        cos_vals (NDArray): Array of cosine similarity values.
        emd_dists (list[float]|None): List of EMD distances. 
        final_metric (NDArray|None): Array of values providing final metric: smallest values -> best fit.
        sort_by (str): Sorting criteria for the images ('cosine', 'emd', 'final'). Must be 'cosine' if at least one of [emd_dists, final_metric] not provided.
    """
    assert sort_by in ['cosine', 'emd', 'final'], f"Invalid sort_by value: {sort_by}. Expected one of ['cosine', 'emd', 'final']"
    assert len(sim_imgs) == len(cos_vals), f"Length of sim_imgs ({len(sim_imgs)}) does not match length of cos_vals ({len(cos_vals)})"
    assert emd_dists is None or len(sim_imgs) == len(emd_dists), f"Length of sim_imgs ({len(sim_imgs)}) does not match length of emd_dists ({len(emd_dists)})"
    assert final_metric is None or len(sim_imgs) == len(final_metric), f"Length of sim_imgs ({len(sim_imgs)}) does not match length of final_metric ({len(final_metric)})"
    assert sort_by == 'cosine' if (emd_dists is None) or (final_metric is None) else True, f"sort_by must be 'cosine' if at least one of [emd_dists, final_metric] is None"
    assert (isinstance(img_path, (list, tuple))) and (all(isinstance(p, str) for p in img_path)) and (len(img_path) in [1,2]), f"img_path must be a str or list of str, got {type(img_path)}"

    sort_mapping = {
        'cosine': 1,
        'emd': 2,
        'final': 3
    }

    if (emd_dists is None) or (final_metric is None):
        sim_list = list(zip(sim_imgs, cos_vals))
    else:
        sim_list = list(zip(sim_imgs, cos_vals, emd_dists, final_metric))

    sim_list.sort(key=lambda x: x[sort_mapping[sort_by]])  # Sort by specified metric

    # Create figure
    if len(img_path) == 1:

        # For single input image
        fig, axs = plt.subplots(1, 6, figsize=(18, 8))
        axs[0].imshow(cv.imread(img_path[0])[..., ::-1])   # BGR to RGB by rearranging channels -> plt reads img as RGB
        axs[0].axis('off')
        axs[0].set_title('Input')
    
    else:
        # For 2 input images
        fig, axs = plt.subplots(2, 6, figsize=(18, 12))

        # Display first input image
        axs[0, 0].imshow(cv.imread(img_path[0])[..., ::-1])   # BGR to RGB by rearranging channels -> plt reads img as RGB
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Input')

        # Display second input image
        axs[1, 0].imshow(cv.imread(img_path[1])[..., ::-1])   # BGR to RGB by rearranging channels -> plt reads img as RGB
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Input')

    # Top 5 Kandidaten (nach EMD sortiert)
    for col, tup in enumerate(sim_list[:5], start=1):
        if (emd_dists is None) or (final_metric is None):
            sim_path, cos_v = tup
            title = f'#{col}\nCos: {cos_v:.3f}'
        else:
            sim_path, cos_v, emd_v, total_score = tup
            title = f'#{col}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}\nTotal Score: {total_score:.3f}'
        axs[0, col].imshow(cv.imread(sim_path)[..., ::-1])
        axs[0, col].set_title(title)
        axs[0, col].axis('off')
        
    plt.tight_layout()
    plt.show()


def calc_emd_and_final(input_vec_l1: NDArray,
                        idx: NDArray, 
                        cosine_values: NDArray, 
                        l1_emb: NDArray, 
                        emd_cost_mat: NDArray|None) -> list[list]:   
        
    """ Calculates the EMD and final metric for the given indices and similar images."""
    emd_values = []
    final_metrics = []

    for i, idx in enumerate(idx):
        # Get complete EMD for 
        db_hist = l1_emb[idx].astype(np.float64)
        dist , _ = emd_with_flow(input_vec_l1, db_hist, distance_matrix=emd_cost_mat)
        emd_values.append(float(dist))
        final_metrics.append(dist / max(cosine_values[i], 1e-9))

    return [emd_values, final_metrics]


def work_that_vectors(input_vec_l1: NDArray,
                    input_vec_l2: NDArray,
                    input_img_paths: list[str],
                    db_img_paths: list[str], 
                    l1_emb: NDArray, 
                    l2_emb: NDArray, 
                    emd_cost_mat: NDArray|None, 
                    batch_size: int=10,
                    track_time: bool=False,
                    show: bool=False) -> NDArray:
    """
    Processes the input image vectors and finds similar images in the database.

    Args:
        input_vec_l1 (NDArray): The L1 feature vector of the input image.
        input_vec_l2 (NDArray): The L2 feature vector of the input image.
        input_img_paths (list[str]): The file paths of the input images.
        db_img_paths (list[str]): The file paths of the database images.
        l1_emb (NDArray): The L1 embeddings of the database images.
        l2_emb (NDArray): The L2 embeddings of the database images.
        emd_cost_mat (NDArray|None): The cost matrix for EMD calculation.
        batch_size (int, optional): The batch size for processing. Defaults to 10.
        track_time (bool, optional): Whether to track processing time and display in terminal. Defaults to False.
        show (bool, optional): Whether to display results in matplotlib window. Defaults to False.

    Returns:
        sim_ordered_paths: Paths of image in backend ordered by similarity as measured within function.
    """
    
    if track_time:
        start = time.time()

    # Calculate cosine similarities
    distances = np.dot(l2_emb, input_vec_l2)

    # Pre-Sort & Find maximum cosine similarites
    t_sorting = time.time()
    filtered_indices, filtered_distances = presort_sims(sims=distances, relative_tresh=.5)

    # Sort remaining imgs
    idx_sorted = np.argsort(filtered_distances)[::-1]       # Sort descending, so that the most similar image is first

    if track_time:
        print(f"Sorting embeddings took: {time.time()-t_sorting:.3f}s")
        previous_batch_start_time = []

    # Init list to return: filepaths in order of similarity as measured down below
    sim_ordered_paths = []

    # Iterate in batches over the sorted indices
    for idx, batch_start in enumerate(range(0, len(idx_sorted), batch_size)):

        if track_time:
            batch_start_time = time.time()
            previous_batch_start_time.append(batch_start_time)

        # Define batch end for better oversight
        batch_end = batch_start + batch_size

        # Collect file paths of the most similar images (according to cosine similarity)
        min_index = filtered_indices[idx_sorted[batch_start:batch_end]]  
        similar_images = [db_img_paths[i] for i in min_index]

        # Get cosine values for the top 10
        cosine_values = filtered_distances[idx_sorted[batch_start:batch_end]]

        if track_time:
            print(f"Cosine time: {time.time()-start:.3f}s") if (batch_start == 0) else None
            start_emd = time.time()

        # For top batchsize candidates, calculate EMD and final metric
        if batch_start == 0:
            emd_values, final_metrics = calc_emd_and_final(input_vec_l1, min_index, cosine_values, l1_emb, emd_cost_mat)
            print(f"EMD time: {time.time()-start_emd:.3f}s\n")
        
        if show:
            if batch_start == 0:
                visualize(input_img_paths, similar_images, cosine_values, emd_values, final_metrics, sort_by='final')
            else:
                visualize(input_img_paths, similar_images, cosine_values, None, None, sort_by='cosine')
    
    # filled list of paths in descending order of similarity
    return sim_ordered_paths


def color_match_single(img_path: list[str], 
                        db_img_paths: list[str], 
                        l1_emb: NDArray, 
                        l2_emb: NDArray, 
                        l_bins: int, a_bins: int, b_bins: int, 
                        emd_cost_mat: NDArray|None, 
                        batch_size: int=10,
                        track_time: bool=False,
                        show: bool=False) -> NDArray:
    
    """Finds the most similar image in the database for a single input image based on cosine similarity and EMD of LAB-space histograms.

    Args:
        img_path (list(str)): List containing the path to the input image. Must contain exactly one image path.
        db_img_paths (list[str]): List of paths to the database images.
        l_bins (int): Number of bins for the L channel.
        a_bins (int): Number of bins for the A channel.
        b_bins (int): Number of bins for the B channel.
        track_time (bool): Whether to track and print the time taken for each step.

    Returns:
        NDArray: Array containing the similarity scores for each database image.
    """
    assert isinstance(img_path, list) and len(img_path) == 1, f"img_path must be a list containing exactly one image path, got {img_path}"

    # Keep track of time if desired
    if track_time:
        start = time.time()

    input_vector_l2 = quantized_image(img_path[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L2')
    input_vector_l1 = quantized_image(img_path[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L1').astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims
    work_that_vectors(input_vec_l1=input_vector_l1, 
                      input_vec_l2=input_vector_l2, 
                      input_img_paths=img_path,
                      db_img_paths=db_img_paths,
                      l1_emb=l1_emb, 
                      l2_emb=l2_emb, 
                      emd_cost_mat=emd_cost_mat,
                      batch_size=batch_size,
                      track_time=track_time,
                      show=show)
    

def color_match_double(img_paths: list[str], 
                        db_img_paths: list[str], 
                        l1_emb: NDArray, 
                        l2_emb: NDArray, 
                        l_bins: int, a_bins: int, b_bins: int, 
                        emd_cost_mat: NDArray|None, 
                        batch_size: int=10,
                        track_time: bool=False,
                        show: bool=False) -> NDArray:

    """Finds the most similar image in the database for a pair of input images based on cosine similarity and EMD of LAB-space histograms.

    Args:
        img_path1 (str): Path to the first input image.
        img_path2 (str): Path to the second input image.
        db_img_paths (list[str]): List of paths to the database images.
        l1_emb (NDArray): L1 embedding matrix.
        l2_emb (NDArray): L2 embedding matrix.
        l_bins (int): Number of bins for the L channel.
        a_bins (int): Number of bins for the A channel.
        b_bins (int): Number of bins for the B channel.
        emd_cost_mat (NDArray|None): Cost matrix for EMD calculation.
        batch_size (int, optional): Size of the batches for processing. Defaults to 10.
        track_time (bool, optional): Whether to track processing time and display in terminal. Defaults to False.
        show (bool, optional): Whether to show visualization in matplotlib. Defaults to False.

    Returns:
        NDArray: Array containing the similarity scores for each database image.
    """
    assert isinstance(img_paths, (list, tuple)) and len(img_paths) == 2, f"img_paths must be a list containing exactly two image paths, got {img_paths}"

    # Extract paths
    img_path1, img_path2 = img_paths

    # Track time if desired
    if track_time:
        start = time.time()

    # Calc L1 and L2 normalized vectors for both images
    combined_vec_l1 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L1').astype(np.float64)

    combined_vec_l2 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L2').astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims
    work_that_vectors(input_vec_l1=combined_vec_l1, 
                      input_vec_l2=combined_vec_l2, 
                      input_img_paths=[img_path1, img_path2],
                      db_img_paths=db_img_paths,
                      l1_emb=l1_emb, 
                      l2_emb=l2_emb, 
                      emd_cost_mat=emd_cost_mat,
                      batch_size=batch_size,
                      track_time=track_time,
                      show=show)
    
    return