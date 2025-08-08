import cv2	
import	numpy as np
from numpy.typing import NDArray



L_BINS = 7
A_BINS = 13
B_BINS = 13



def quantized_image(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    """
    Function to extract color signature from an image and map it to quantized LAB colors.

    Args:
        filepath (str): Path to the image file
        l_bins (int): Amount of bins on lightness-channel
        a_bins (int): Amount of bins on a-channel
        b_bins (int): Amount of bins on b-channel
        normalization (str): Type of normalization to apply ('L1' or 'L2')

    Returns:
        histogram_vector (NDArray): Normalized histogram vector of quantized colors in origian CIE-LAB 
    """
    # Check if the normalization parameter is valid
    if normalization not in ['L1', 'L2']:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")

    img = cv2.imread(filepath)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Median-Filter auf a und b
    l, a, b = cv2.split(lab_img)
    a = cv2.medianBlur(a, ksize=3)
    b = cv2.medianBlur(b, ksize=3)

    # In echten CIE-LAB Farbraum konvertieren
    cie_lab = cv_to_cie(cv2.merge([l, a, b]))

    # Flach machen
    pixels = cie_lab.reshape(-1, 3)
    l, a, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(-128, 127, a_bins + 1)
    b_edges = np.linspace(-128, 127, b_bins + 1)

    l_idx = np.clip(np.digitize(l, l_edges) - 1, 0, l_bins - 1)
    a_idx = np.clip(np.digitize(a, a_edges) - 1, 0, a_bins - 1)
    b_idx = np.clip(np.digitize(b, b_edges) - 1, 0, b_bins - 1)

    linear_idx = l_idx * (a_bins * b_bins) + a_idx * b_bins + b_idx   

    total_bins = l_bins * a_bins * b_bins
    histogram_vector = np.bincount(linear_idx, minlength=total_bins).astype(np.float32)


    if normalization == 'L1':
        histogram_vector /= np.sum(histogram_vector)
    else:   
        # L2 normalization
        histogram_vector /= max(np.linalg.norm(histogram_vector), 1e-12)
    return histogram_vector


def get_bin_centers(l_bins: int, a_bins: int, b_bins: int) -> np.ndarray:
    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(-128, 127, a_bins + 1)
    b_edges = np.linspace(-128, 127, b_bins + 1)

    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2

    grid = np.array(np.meshgrid(l_centers, a_centers, b_centers, indexing='ij'))
    return grid.reshape(3, -1).T


def quantized_image_signed(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    img = cv2.imread(filepath)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    a_denoised = cv2.medianBlur(a, ksize=3)
    b_denoised = cv2.medianBlur(b, ksize=3)
    cie_lab = cv_to_cie(cv2.merge([l, a_denoised, b_denoised]))
    l_vals, a_vals, b_vals = cie_lab[:, :, 0].flatten(), cie_lab[:, :, 1].flatten(), cie_lab[:, :, 2].flatten()

    # set bin-edges
    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(0, 128, a_bins + 1)   # half a-dim since we combine neg-colors into 1 dim
    b_edges = np.linspace(0, 128, b_bins + 1)   

    total_bins = l_bins * a_bins * b_bins

    # Masks to select pixel that "phase out" one another
    masks = {
        '+a +b': (a_vals >= 0) & (b_vals >= 0),
        '-a -b': (a_vals <  0) & (b_vals <  0),
        '+a -b': (a_vals >= 0) & (b_vals <  0),
        '-a +b': (a_vals <  0) & (b_vals >= 0),
    }

    # 
    all_idxs = []
    all_weights = []

    for label, mask in masks.items():
        l_bin = l_vals[mask]
        if label == '+a +b':
            a_bin = a_vals[mask]
            b_bin = b_vals[mask]
            weight = +1
        elif label == '-a -b':
            a_bin = -a_vals[mask]
            b_bin = -b_vals[mask]
            weight = -1
        elif label == '+a -b':
            a_bin = a_vals[mask]
            b_bin = -b_vals[mask]
            weight = +1
        elif label == '-a +b':
            a_bin = -a_vals[mask]
            b_bin = b_vals[mask]
            weight = -1

        l_idx = np.clip(np.digitize(l_bin, l_edges) - 1, 0, l_bins - 1)
        a_idx = np.clip(np.digitize(a_bin, a_edges) - 1, 0, a_bins - 1)
        b_idx = np.clip(np.digitize(b_bin, b_edges) - 1, 0, b_bins - 1)

        # Linearize
        linear_idx = l_idx * (a_bins * b_bins) + a_idx * b_bins + b_idx
        all_idxs.append(linear_idx)
        all_weights.append(np.full_like(linear_idx, weight, dtype=np.float32))

    # Combine
    full_idx = np.concatenate(all_idxs)
    full_weights = np.concatenate(all_weights)
    signed_histogram = np.bincount(full_idx, weights=full_weights, minlength=total_bins)

    # Normalization
    if normalization == 'L1':
        norm = np.sum(np.abs(signed_histogram))
    else:
        norm = np.linalg.norm(signed_histogram, ord=2)  # L2

    if norm > 0:
        signed_histogram /= norm

    return signed_histogram


def quantized_image_two_vecs(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    # Check if the normalization parameter is valid
    if normalization not in ['L1', 'L2']:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")

    img = cv2.imread(filepath)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Median-Filter auf a und b
    l, a, b = cv2.split(lab_img)
    a = cv2.medianBlur(a, ksize=3)
    b = cv2.medianBlur(b, ksize=3)

    # In echten CIE-LAB Farbraum konvertieren
    cie_lab = cv_to_cie(cv2.merge([l, a, b]))

    # Flach machen
    pixels = cie_lab.reshape(-1, 3)
    l, a, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(-128, 127, a_bins + 1)
    b_edges = np.linspace(-128, 127, b_bins + 1)

    l_idx = np.clip(np.digitize(l, l_edges) - 1, 0, l_bins - 1)
    a_idx = np.clip(np.digitize(a, a_edges) - 1, 0, a_bins - 1)
    b_idx = np.clip(np.digitize(b, b_edges) - 1, 0, b_bins - 1)

    linear_idx_ab = a_idx * b_bins + b_idx  
    linear_idx_l = l_idx 

    total_bins_ab = a_bins * b_bins
    histogram_vector_ab = np.bincount(linear_idx_ab, minlength=total_bins_ab).astype(np.float32)
    histogram_vector_l = np.bincount(linear_idx_l, minlength=l_bins).astype(np.float32)

    if normalization == 'L1':
        histogram_vector_ab /= np.sum(histogram_vector_ab)
        histogram_vector_l /= np.sum(histogram_vector_l)
    else:   
        # L2 normalization
        histogram_vector_ab /= np.linalg.norm(histogram_vector_ab, ord=2)
        histogram_vector_l /= np.linalg.norm(histogram_vector_l, ord=2)

    return histogram_vector_ab, histogram_vector_l



def cv_to_cie(img: NDArray) -> NDArray:
    """Converts image in LAB-space from values ranges [[0,255], [0,255], [0,255]] (as in openCV) to [[0,100], [-128, 127], [-128, 127]] (the real CIE-LAB)
    
    Args:
        img (NDArray): image in LAB-color space with range [[0,255], [0,255], [0,255]]
        
    Returns:
        cie-img (NDArray): image with corrected value-range to match CIE-LAB space
    """
    l, a, b = cv2.split(img)
    l = (l.astype(np.float32) / 255.0) * 100.0
    a = a.astype(np.float32) - 128.0
    b = b.astype(np.float32) - 128.0
    return cv2.merge([l, a, b])