import cv2	
import	numpy as np
from numpy.typing import NDArray



L_BINS = 13
A_BINS = 5
B_BINS = 5



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
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    a_denoised = cv2.medianBlur(a, ksize=3)
    b_denoised = cv2.medianBlur(b, ksize=3)
    cie_lab = cv_to_cie(img=cv2.merge([l, a_denoised, b_denoised]))
    l, a, b = cie_lab[:,:,0], cie_lab[:,:,1], cie_lab[:,:,2]
    denoised = cv2.merge([l, a_denoised, b_denoised])
    # blurred = ski.filters.gaussian(img, sigma=1.4, truncate=2.5)
    # blurred = (denoised * 255).astype(np.uint8) 
    pixels = denoised.reshape(-1, 3)
    l, a, b = pixels[:,0], pixels[:,1], pixels[:,2]

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
        histogram_vector /= np.linalg.norm(histogram_vector, ord=2)

    return histogram_vector


def get_bin_centers(l_bins: int, a_bins: int, b_bins: int, round: bool) -> NDArray:
    """
    Return center-colors for each bin
    """
    if round:
        l_centers = np.round(np.linspace(-128, 127, l_bins))
        a_centers = np.round(np.linspace(-128, 127, a_bins))
        b_centers = np.round(np.linspace(-128, 127, b_bins))
    else: 
        l_centers = np.linspace(-128, 127, l_bins)
        a_centers = np.linspace(-128, 127, a_bins)
        b_centers = np.linspace(-128, 127, b_bins)

    grid = np.array(np.meshgrid(l_centers, a_centers, b_centers, indexing='ij'))
    bin_centers = grid.reshape(3, -1).T  # shape: (total_bins, 3)
    return bin_centers


def quantized_image_signed(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    img = cv2.imread(filepath)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    a_denoised = cv2.medianBlur(a, ksize=3)
    b_denoised = cv2.medianBlur(b, ksize=3)
    cie_lab = cv_to_cie(cv2.merge([l, a_denoised, b_denoised]))
    l_vals, a_vals, b_vals = cie_lab[:, :, 0].flatten(), cie_lab[:, :, 1].flatten(), cie_lab[:, :, 2].flatten()

    # Symmetrisches L: Spiegelung um L=50
    l_sym = np.minimum(l_vals, 100 - l_vals)

    # Bin-Kanten definieren
    l_edges = np.linspace(0, 50, l_bins + 1)
    a_edges = np.linspace(0, 128, a_bins + 1)
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
        l_bin = l_sym[mask]
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


def quantized_image_three_vecs(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    img = cv2.imread(filepath)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    a_denoised = cv2.medianBlur(a, ksize=3)
    b_denoised = cv2.medianBlur(b, ksize=3)
    denoised = cv2.merge([l, a_denoised, b_denoised])
    # blurred = ski.filters.gaussian(img, sigma=1.4, truncate=2.5)
    # blurred = (denoised * 255).astype(np.uint8) 
    pixels = denoised.reshape(-1, 3)

    # Make every axis range from -128 to 127
    pixels = pixels - 128   # L,a & b axis now have pos. and neg. part

    # Create positive bins 
    l_bins_edges = np.linspace(-128, 127, l_bins + 1)
    a_bins_edges = np.linspace(-128, 127, a_bins + 1)
    b_bins_edges = np.linspace(-128, 127, b_bins + 1)

    # Extrahiere die Pixelwerte
    l_vals = pixels[:, 0]
    a_vals = pixels[:, 1]
    b_vals = pixels[:, 2]

    # Positive bins: mask for positive values
    pos_mask = (l_vals >= 0) & (a_vals >= 0) & (b_vals >= 0)
    neg_mask = (l_vals < 0) & (a_vals < 0) & (b_vals < 0)

    # Bin indices für positive und negative Teile
    l_idx_pos = np.digitize(l_vals[pos_mask], l_bins_edges) - 1
    a_idx_pos = np.digitize(a_vals[pos_mask], a_bins_edges) - 1
    b_idx_pos = np.digitize(b_vals[pos_mask], b_bins_edges) - 1

    l_idx_neg = np.digitize(-l_vals[neg_mask], l_bins_edges) - 1
    a_idx_neg = np.digitize(-a_vals[neg_mask], a_bins_edges) - 1
    b_idx_neg = np.digitize(-b_vals[neg_mask], b_bins_edges) - 1

    l_idx_all = np.digitize(l_vals, l_bins_edges) - 1

    l_idx_pos = np.clip(l_idx_pos, 0, l_bins-1)
    a_idx_pos = np.clip(a_idx_pos, 0, a_bins-1)
    b_idx_pos = np.clip(b_idx_pos, 0, b_bins-1)

    l_idx_neg = np.clip(l_idx_neg, 0, l_bins-1)
    a_idx_neg = np.clip(a_idx_neg, 0, a_bins-1)
    b_idx_neg = np.clip(b_idx_neg, 0, b_bins-1)

    l_idx_all = np.clip(l_idx_all, 0, l_bins-1)


    # Linear indices for bincount
    total_bins = (l_bins * a_bins * b_bins)
    linear_idx_pos = l_idx_pos * (a_bins * b_bins) + a_idx_pos * b_bins + b_idx_pos
    linear_idx_neg = l_idx_neg * (a_bins * b_bins) + a_idx_neg * b_bins + b_idx_neg

    # Linear index stays the same for l_idx, since it's only one channel

    # Histogram for positive, negative, lightness
    hist_pos = np.bincount(linear_idx_pos, minlength=total_bins).astype(np.float32)     
    hist_neg = np.bincount(linear_idx_neg, minlength=total_bins).astype(np.float32)
    hist_light = np.bincount(l_idx_all).astype(np.float64)   # No minlength needed, does not get compared with hists of different kind


    # Normalization as desired
    if normalization == 'L1':
        norm_pos = np.sum(np.abs(hist_pos))
        norm_neg = np.sum(np.abs(hist_neg))
        norm_light = np.sum(np.abs(hist_light))

        if norm_pos > 0:
            hist_pos /= norm_pos
        else:
            hist_pos[:] = 0

        if norm_neg > 0:
            hist_neg /= norm_neg
        else:
            hist_neg[:] = 0

        if norm_light > 0:
            hist_light /= hist_light
        else:
            hist_light[:] = 0
    
    # L2 normalization
    else:
        norm_pos = np.linalg.norm(hist_pos, ord=2)
        norm_neg = np.linalg.norm(hist_neg, ord=2)
        norm_light = np.linalg.norm(hist_light, ord=2)

        if norm_pos > 0:
            hist_pos /= norm_pos
        else:
            hist_pos[:] = 0

        if norm_neg > 0:
            hist_neg /= norm_neg
        else:
            hist_neg[:] = 0

        if norm_light > 0:
            hist_light /= norm_light
        else:
            hist_light[:] = 0

    return hist_pos, hist_neg, hist_light

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