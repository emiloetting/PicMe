import cv2	
import	numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from annoy import AnnoyIndex
from itertools import product
from typing import Tuple
from skimage.color import lab2rgb



cwd = os.getcwd()
image = "my_test_file.jpg"
filepath  = os.path.join(cwd, 'ColorSimilarity',  image)
filepath1 = os.path.join(cwd, 'ColorSimilarity', "Schwan.jpeg")
filepath2 = os.path.join(cwd, 'ColorSimilarity', "Schwan.jpeg")



def extract_color_signature(img_path: str, n_clusters: int) -> tuple:
    """
    Extracts the color signature of an image using k-means clustering in the Lab color space.
    The function normalizes the pixel values, applies k-means clustering, and returns the cluster centers and their weights.

    The cluster centers are converted back to BGR color space for visualization.
    The weights are normalized to sum to 1.

    Arguments:
        img_path [str]: Path to the image file.

    Returns:
        [centers (LAB) [np.array], weights[np.array]]
    """
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    pixels = np.float32(img.reshape(-1, 3))

    # normalise the pixel values
    # pixels[:, 0] /= 100.0  
    # pixels[:, 1:] += 128.0  
    # pixels[:,:] /= 255.0	   

    k=n_clusters

    # criteria to stop the algorithm (max. 100 iterations, 0.1 epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # k clusters, no labels, criteria from above, 10 runs with different centers, cluster centers
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # calculate weights
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(k)
    for i, label in enumerate(unique_labels):
        weights[label] = counts[i]

    # normalize weights
    weights = weights / np.sum(weights)  

    
    return [centers, weights]


def visualize_color_signature(centers, weights, figsize=(10, 4)):
    """
    Visualisiert eine Farbsignatur.
    
    Parameters:
    -----------
    centers : numpy.ndarray
        Die k repräsentativen Farben
    weights : numpy.ndarray
        Die relative Häufigkeit jeder Farbe
    """
    sorted_indices = np.argsort(weights)[::-1]
    sorted_centers = centers[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    plt.figure(figsize=figsize)
    
    for i, (center, weight) in enumerate(zip(sorted_centers, sorted_weights)):
        center[0] /= 255.0
        center[0] *= 100.0
        center[1:] -= 128.0
        
        rgb_color = lab2rgb(center)

        plt.bar(i, weight, color=rgb_color, width=0.8)
        
        plt.text(i, weight + 0.01, f"{weight*100:.1f}%", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Farbcluster')
    plt.ylabel('Relative Häufigkeit')
    plt.title('Farbsignatur des Bildes')
    plt.xticks([])  
    plt.ylim(0, max(weights) * 1.2)  
    plt.show()


def compare_images(weights1, centers1, weights2, centers2):
    """
    Vergleicht zwei Farbsignaturen mithilfe der Earth Mover's Distance (EMD).

    Die Funktion berechnet die EMD zwischen zwei Farbsignaturen, die durch ihre Gewichte und Zentren dargestellt werden.
    Die Gewichte und Zentren werden in numpy-Arrays umgewandelt, bevor die EMD berechnet wird.

    Arguments:
    
            weights1 [np.array]: Gewichte der ersten Farbsignatur.
            centers1 [np.array]: Zentren der ersten Farbsignatur.
            weights2 [np.array]: Gewichte der zweiten Farbsignatur.
            centers2 [np.array]: Zentren der zweiten Farbsignatur.

    Returns:
            [distance [float]]: Die berechnete EMD zwischen den beiden Farbsignaturen.
    """
    weights1 = np.array(weights1, dtype=np.float32)
    weights2 = np.array(weights2, dtype=np.float32)
    centers1 = np.array(centers1, dtype=np.float32)
    centers2 = np.array(centers2, dtype=np.float32)


    signature1 = np.column_stack((weights1, centers1))
    signature2 = np.column_stack((weights2, centers2))


    distance, _,_ = cv2.EMD(signature1,signature2,cv2.DIST_L1)

    return distance

def get_quantized_LAB(l_bins: int, a_bins: int, b_bins: int) -> Tuple[np.ndarray, int]:
    """
    Erzeugt eine quantisierte LAB-Farbtabelle.

    Parameters:
    -----------
    l_bins : int
        Anzahl der L-Bins
    a_bins : int
        Anzahl der a-Bins
    b_bins : int
        Anzahl der b-Bins

    Returns:
    --------
    quantized_lab : numpy.ndarray
        Quantisierte LAB-Farbtabelle
    """
    # OpenCV‑Lab value ranges: L∈[0,255], a∈[0,255], b∈[0,255]
    l = np.linspace(0, 255, l_bins, dtype=np.uint8)
    a = np.linspace(0, 255, a_bins, dtype=np.uint8)
    b = np.linspace(0, 255, b_bins, dtype=np.uint8)

    quantized_lab = np.array(list(product(l, a, b)))    # build cartesian product
    quantized_lab = quantized_lab.astype(np.uint8)

    amount_colors = quantized_lab.shape[0]
    return quantized_lab, amount_colors

def quantized_image(filepath: str, annoy_index: AnnoyIndex, quantized_cosine_LAB: np.array, normalization: str) -> np.array:
    """
    Function to extract color signature from an image and map it to quantized LAB colors.

    Parameters:
    - filepath: Path to the image file.
    - quantized_cosine_LAB_tree: cKDTree for fast nearest neighbor search.
    - quantized_cosine_LAB: Array of quantized LAB colors.
    - normalization: Type of normalization to apply ('L1' or 'L2').

    Returns:
    - histogram_vector (np.array): Normalized histogram vector of quantized colors.
    """
    # Check if the normalization parameter is valid
    if normalization not in ['L1', 'L2']:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")
    
    # Extract color signatures for the images
    centers, weights = extract_color_signature(filepath, n_clusters=64)  # 64 colors for cosine quantization step

    # Map the color signature to the quantized LAB colors
    quantized_indices = [annoy_index.get_nns_by_vector(center.astype(np.float32).tolist(), 1)[0] for center in centers]   # Get the indices of the nearest quantized colors
    quantized_indices = np.array(quantized_indices, dtype=np.int32)  # Convert to numpy array

    # In case 2 centers were quantized to the same color: remove duplicates, adapt weights
    unique_ids, inverse = np.unique(
        quantized_indices,
        axis=0,
        return_inverse=True
        )

    aggregated_weights = np.bincount(inverse, weights=weights)

    # Create histogram vector of quantized colors
    histogram_vector = np.zeros(quantized_cosine_LAB.shape[0], dtype=np.float32)
    histogram_vector[unique_ids] = aggregated_weights

    if normalization == 'L1':
        histogram_vector /= np.sum(histogram_vector)
    else:   
        # L2 normalization
        histogram_vector /= np.linalg.norm(histogram_vector, ord=2)

    return histogram_vector

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Function to calculate the cosine similarity between two vectors.
    """
    norm_a = np.linalg.norm(vector1)    # L2 normalize vectos
    norm_b = np.linalg.norm(vector2)
    if norm_a == 0 or norm_b == 0:   # Avoid division by zero
        return 0.0
    return np.dot(vector1, vector2) / (norm_a * norm_b)



if __name__ == "__main__":
    centers, weights = extract_color_signature(filepath)
    visualize_color_signature(centers, weights)
    # centers1, weights1 = extract_color_signature(filepath1)
    # centers2, weights2 = extract_color_signature(filepath2)
    # dauer = timeit.timeit(lambda : compare_images(weights1, centers1, weights2, centers2), number=500_000)
    # print(dauer)

