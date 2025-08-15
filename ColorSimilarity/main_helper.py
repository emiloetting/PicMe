import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cv2 as cv
from colorClusterquantized import *
import time
from pyemd import emd_with_flow
import annoy as ann
import os



ann_idx_path = os.path.join(os.getcwd(), 'ColorSimilarity', 'color_index_l2.ann')
ann_idx = ann.AnnoyIndex(L_BINS*A_BINS*B_BINS, 'angular')
ann_idx.load(ann_idx_path)


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


def visualize(img_path: list[str], sim_imgs: list, cos_vals: NDArray, emd_dists: list[float]|None, final_metric: NDArray|None, sort_by: str, img_weights: list[float]|None=None):
    """Opens matplotlib visualization of top-12 images based on cosine similarities and EMD histograms.

    Args:
        img_path (str|list[str]): Path to the input image or list of image paths.
        sim_imgs (list): List of similar images.
        cos_vals (NDArray): Array of cosine similarity values.
        emd_dists (list[float]|None): List of EMD distances. 
        final_metric (NDArray|None): Array of values providing final metric: smallest values -> best fit.
        sort_by (str): Sorting criteria for the images ('cosine', 'emd', 'final'). Must be 'cosine' if at least one of [emd_dists, final_metric] not provided.
        img_weights (list[float]|None): Weights for each image's contribution to the final metric. Must sum to 2.
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

    # Festlegen der Größe basierend auf Anzahl der Input-Bilder
    if len(img_path) == 1:
        # Ein einzelnes Input-Bild
        fig, axs = plt.subplots(2, 7, figsize=(24, 10))
        
        # Input-Bild anzeigen
        axs[0, 0].imshow(cv.imread(img_path[0])[..., ::-1])
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Input')
        
        # Leere Stelle für die Ausrichtung
        axs[1, 0].axis('off')
        
        # Top-12 Bilder anzeigen (6 pro Zeile)
        for i, tup in enumerate(sim_list[:12]):
            row = i // 6
            col = i % 6 + 1
            
            if (emd_dists is None) or (final_metric is None):
                sim_path, cos_v = tup
                title = f'#{i+1}\nCos: {cos_v:.3f}'
            else:
                sim_path, cos_v, emd_v, total_score = tup
                title = f'#{i+1}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}\nScore: {total_score:.3f}'
            
            axs[row, col].imshow(cv.imread(sim_path)[..., ::-1])
            axs[row, col].set_title(title)
            axs[row, col].axis('off')
    
    else:
        # Zwei Input-Bilder
        fig, axs = plt.subplots(3, 7, figsize=(24, 15))
        
        # Erstes Input-Bild anzeigen
        axs[0, 0].imshow(cv.imread(img_path[0])[..., ::-1])
        axs[0, 0].axis('off')
        if img_weights is not None:
            axs[0, 0].set_title(f'Input 1\nWeight: {img_weights[0]}')
        else:
            axs[0, 0].set_title('Input 1')
        
        # Zweites Input-Bild anzeigen
        axs[1, 0].imshow(cv.imread(img_path[1])[..., ::-1])
        axs[1, 0].axis('off')
        if img_weights is not None:
            axs[1, 0].set_title(f'Input 2\nWeight: {img_weights[1]}')
        else:
            axs[1, 0].set_title('Input 2')
            
        # Leere Stelle für die Ausrichtung
        axs[2, 0].axis('off')
        
        # Top-12 Bilder anzeigen (6 pro Zeile)
        for i, tup in enumerate(sim_list[:12]):
            row = i // 6
            col = i % 6 + 1
            
            if (emd_dists is None) or (final_metric is None):
                sim_path, cos_v = tup
                title = f'#{i+1}\nCos: {cos_v:.3f}'
            else:
                sim_path, cos_v, emd_v, total_score = tup
                title = f'#{i+1}\nCos: {cos_v:.3f}\nEMD: {emd_v:.3f}\nScore: {total_score:.3f}'
            
            axs[row, col].imshow(cv.imread(sim_path)[..., ::-1])
            axs[row, col].set_title(title)
            axs[row, col].axis('off')
    
    # Leere Achsen ausblenden
    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            if not axs[row, col].has_data():
                fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.show()


def calc_emd_and_final(input_vec_l1: NDArray,
                        idx: NDArray, 
                        cosine_values: NDArray, 
                        l1_emb: NDArray, 
                        emd_cost_mat: NDArray|None) -> list[list]:   
        
    """ Calculates the EMD and final metric for the given indices and similar images.

    Args:
        input_vec_l1 (NDArray): The L1 feature vector of the input image.
        idx (NDArray): The indices of the similar images.
        cosine_values (NDArray): The cosine similarity values of the similar images.
        l1_emb (NDArray): The L1 embeddings of the database images.
        emd_cost_mat (NDArray|None): The cost matrix for EMD calculation.

    Returns:
        list[list]: A list containing the EMD values and final metrics.
    """
    assert input_vec_l1.ndim == 1, f"Expected (n,), got {input_vec_l1.shape}"
    assert idx.ndim == 1, f"Expected (n,), got {idx.shape}"
    assert cosine_values.ndim == 1, f"Expected (n,), got {cosine_values.shape}"
    assert l1_emb.ndim == 2, f"Expected (n, m), got {l1_emb.shape}"
    assert emd_cost_mat is None or emd_cost_mat.ndim == 2, f"Expected (n, n), got {emd_cost_mat.shape if emd_cost_mat is not None else None}"

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
                    img_weights: list[float]|None=None,
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
        img_weights (list[float]|None): Weights for each image's contribution to the final metric. Must sum to 2 and each weight must be positive.
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
                visualize(input_img_paths, similar_images, cosine_values, emd_values, final_metrics, sort_by='final', img_weights=img_weights)
            else:
                visualize(input_img_paths, similar_images, cosine_values, None, None, sort_by='cosine', img_weights=img_weights)

    # filled list of paths in descending order of similarity
    return sim_ordered_paths


def work_that_vectors_ann(input_vec_l1: NDArray,
                    input_vec_l2: NDArray,
                    input_img_paths: list[str],
                    db_img_paths: list[str], 
                    l1_emb: NDArray, 
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

    # Paths in initial order (by cosine similarity)
    sim_ordered_paths = [db_img_paths[index] for index in indices]

    if track_time:
        print(f"Annoy search time: {time.time()-start:.3f}s")
        start_emd = time.time()

    # Berechne EMD für die angegebene Anzahl von Bildern
    # Nutze min() um sicherzustellen, dass wir nicht mehr Bilder verarbeiten als wir haben
    emd_calc_size = min(emd_count, len(indices))
    top_indices = indices[:emd_calc_size]
    top_cosine_values = cosine_values[:emd_calc_size]
    
    emd_values, final_metrics = calc_emd_and_final(
        input_vec_l1, 
        top_indices, 
        top_cosine_values, 
        l1_emb, 
        emd_cost_mat
    )
    
    # Sortiere die Bilder mit EMD-Berechnung nach final_metrics
    top_paths = [db_img_paths[i] for i in top_indices]
    sorted_indices = np.argsort(final_metrics)  # sort ascending by final metrics
    sorted_top_paths = [top_paths[i] for i in sorted_indices]
    
    # Ersetze die entsprechenden Einträge in sim_ordered_paths mit den sortierten
    sim_ordered_paths[:emd_calc_size] = sorted_top_paths
    
    if track_time and emd_calc_size > 0:
        print(f"EMD time: {time.time()-start_emd:.3f}s\n")
    
    if show:
        visualize(
            input_img_paths, 
            sim_ordered_paths[:12],  # 12 since we have 12 spaces in the visualization
            cosine_values[:12],      # 12 since we have 12 spaces in the visualization
            emd_values,
            final_metrics,
            sort_by='final', 
            img_weights=img_weights
        )

    # Liste der Pfade in absteigender Ähnlichkeit
    return sim_ordered_paths  # Return only the top 12 results for visualization


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
        l1_emb (NDArray): L1 embedding matrix.
        l2_emb (NDArray): L2 embedding matrix.
        l_bins (int): Number of bins for the L channel.
        a_bins (int): Number of bins for the A channel.
        b_bins (int): Number of bins for the B channel.
        emd_cost_mat (NDArray|None): Cost matrix for EMD calculation.
        batch_size (int): Batch size for processing.
        track_time (bool): Whether to track and print the time taken for each step.

    Returns:
        NDArray: Array containing the similarity scores for each database image.
    """
    assert isinstance(img_path, list) and len(img_path) == 1, f"img_path must be a list containing exactly one image path, got {img_path}"
    assert isinstance(db_img_paths, list) and len(db_img_paths) > 0, f"db_img_paths must be a non-empty list, got {db_img_paths}"
    assert isinstance(l1_emb, np.ndarray) and l1_emb.ndim == 2, f"l1_emb must be a 2D numpy array, got {l1_emb.shape if l1_emb is not None else None}"
    assert isinstance(l2_emb, np.ndarray) and l2_emb.ndim == 2, f"l2_emb must be a 2D numpy array, got {l2_emb.shape if l2_emb is not None else None}"
    assert isinstance(emd_cost_mat, np.ndarray) and emd_cost_mat.ndim == 2, f"emd_cost_mat must be a 2D numpy array, got {emd_cost_mat.shape if emd_cost_mat is not None else None}"
    assert isinstance(batch_size, int) and batch_size > 0, f"batch_size must be a positive integer, got {batch_size}"
    assert isinstance(track_time, bool), f"track_time must be a boolean, got {type(track_time)}"
    assert isinstance(show, bool), f"show must be a boolean, got {type(show)}"

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
    
def color_match_single_ann(img_path: list[str], 
                        db_img_paths: list[str], 
                        l1_emb: NDArray, 
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

    input_vector_l2 = quantized_image(img_path[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L2')
    input_vector_l1 = quantized_image(img_path[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization='L1', adjusted_bin_size=adjusted_bin_size).astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims and return the result
    return work_that_vectors_ann(input_vec_l1=input_vector_l1, 
                      input_vec_l2=input_vector_l2, 
                      input_img_paths=img_path,
                      db_img_paths=db_img_paths,
                      l1_emb=l1_emb, 
                      annoy_index=annoy_index, 
                      emd_cost_mat=emd_cost_mat,
                      num_results=num_results,  # Use batch_size as number of results
                      emd_count=emd_count,  # Use batch_size for EMD calculations
                      track_time=track_time,
                      show=show)
    

def color_match_double(img_paths: list[str], 
                        db_img_paths: list[str], 
                        l1_emb: NDArray, 
                        l2_emb: NDArray, 
                        l_bins: int, a_bins: int, b_bins: int, 
                        emd_cost_mat: NDArray|None, 
                        img_weights: list[float]=[1.0, 1.0],
                        num_results: int=12,
                        emd_count: int=12,
                        track_time: bool=False,
                        show: bool=False) -> NDArray:

    """Finds the most similar image in the database for a pair of input images based on cosine similarity and EMD of LAB-space histograms.

    Args:
        img_paths (list[str]): List containing the paths to the input images. Must contain exactly two image paths.
        db_img_paths (list[str]): List of paths to the database images.
        l1_emb (NDArray): L1 embedding matrix.
        l2_emb (NDArray): L2 embedding matrix.
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
    img_path1, img_path2 = img_paths

    # Track time if desired
    if track_time:
        start = time.time()

    # Calc L1 and L2 normalized vectors for both images
    combined_vec_l1 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L1', weights=img_weights).astype(np.float64)

    combined_vec_l2 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L2', weights=img_weights).astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims
    return work_that_vectors_ann(input_vec_l1=combined_vec_l1, 
                      input_vec_l2=combined_vec_l2, 
                      input_img_paths=[img_path1, img_path2],
                      db_img_paths=db_img_paths,
                      l1_emb=l1_emb, 
                      l2_emb=l2_emb, 
                      emd_cost_mat=emd_cost_mat,
                      num_results=num_results,  # Use batch_size as number of results
                      emd_count=emd_count,  # Use batch_size for EMD calculations
                      track_time=track_time,
                      show=show)

def color_match_double_ann(img_paths: list[str], 
                        db_img_paths: list[str], 
                        l1_emb: NDArray, 
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
    img_path1, img_path2 = img_paths

    # Track time if desired
    if track_time:
        start = time.time()

    # Calc L1 and L2 normalized vectors for both images
    combined_vec_l1 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L1', adjusted_bin_size=adjusted_bin_size, weights=img_weights).astype(np.float64)

    combined_vec_l2 = quantize2images(filepaths=[img_path1, img_path2],
                                      l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, 
                                      normalization='L2', adjusted_bin_size=False, weights=img_weights).astype(np.float64)

    if track_time:
        print(f'Hists for input took: {time.time()-start:.3f}s')
        start = time.time()

    # Calc sims
    return work_that_vectors_ann(input_vec_l1=combined_vec_l1, 
                      input_vec_l2=combined_vec_l2, 
                      input_img_paths=[img_path1, img_path2],
                      db_img_paths=db_img_paths,
                      l1_emb=l1_emb, 
                      annoy_index=annoy_index, 
                      emd_cost_mat=emd_cost_mat,
                      num_results=num_results,  # Use batch_size as number of results
                      emd_count=emd_count,  # Use batch_size for EMD calculations
                      track_time=track_time,
                      show=show)