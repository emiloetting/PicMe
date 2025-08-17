import cv2	
import	numpy as np
from numpy.typing import NDArray



def quantized_image(img: NDArray, l_bins: int, a_bins: int, b_bins: int, normalization: str|None, adjusted_bin_size: bool=False) -> NDArray:
    """
    Function to extract color signature from an image and map it to quantized LAB colors.

    Args:
        filepath (str): Path to the image file
        l_bins (int): Amount of bins on lightness-channel
        a_bins (int): Amount of bins on a-channel
        b_bins (int): Amount of bins on b-channel
        normalization (str|None): Type of normalization to apply ('L1' or 'L2')
        adjusted_bin_size (bool): Whether to adjust bin sizes for faster EMD calculation when normalization is 'L1'

    Returns:
        histogram_vector (NDArray): Normalized histogram vector of quantized colors in origian CIE-LAB 
    """
    assert not (normalization != 'L1' and adjusted_bin_size), f"Adjusted bin size can only be used with normalization 'L1', not '{normalization}'"
    # Check if the normalization parameter is valid
    if normalization not in ['L1', 'L2', None]:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")

    rsz_img = downscale_to_fix_size(img=img, max_pixels=1_000_000)
    lab_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2Lab)

    # For faster emd-calc: smaller bins for L1 normalization
    if normalization == 'L1' and adjusted_bin_size:
        l_bins = max(1, round(l_bins/1.3))  # Ensure at least 1 bin
        a_bins = max(1, round(a_bins/1.3))
        b_bins = max(1, round(b_bins/1.3))

    cie_lab = cv_to_cie(lab_img)
    
    # flatten
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
    elif normalization == 'L2':   
        histogram_vector /= max(np.linalg.norm(histogram_vector), 1e-12)

    # If normalization == None just return unnormalized vector    
    return histogram_vector


def get_bin_centers(l_bins: int, a_bins: int, b_bins: int) -> np.ndarray:
    """Returns the centers of the bins in CIE-LAB color space.

    Args:
        l_bins (int): Number of bins in L channel.
        a_bins (int): Number of bins in A channel.
        b_bins (int): Number of bins in B channel.

    Returns:
        np.ndarray: Array of shape (n_bins, 3) containing the center coordinates of each bin.
    """
    assert l_bins > 0 and a_bins > 0 and b_bins > 0, "Number of bins must be greater than 0"
    assert isinstance(l_bins, int) and isinstance(a_bins, int) and isinstance(b_bins, int), "Number of bins must be integers"

    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(-128, 127, a_bins + 1)
    b_edges = np.linspace(-128, 127, b_bins + 1)

    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2

    grid = np.array(np.meshgrid(l_centers, a_centers, b_centers, indexing='ij'))
    return grid.reshape(3, -1).T


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


def downscale_to_fix_size(img: NDArray, max_pixels: int = 250_000) -> NDArray:
    """
    Scales image to fixed size of max_pixels while keeping aspect ratio.
    
    Args:
        img (NDArray): input image to be scaled
        max_pixels (int): max amount of pixel for image to hold after scaling

    Returns:
        NDArray: possibly downscaled image
    """
    assert isinstance(img, np.ndarray), f"Input must be a numpy array, not {type(img)}"
    assert img.ndim in [2, 3], f"Input image must be 2D or 3D, not {img.ndim}D"
    assert img.size > 0, "Input image must not be empty"
    assert isinstance(max_pixels, int) and max_pixels > 0, f"max_pixels must be a positive integer, not {max_pixels}"

    h, w = img.shape[:2]
    curr_pixels = h * w

    # Early return with no changes made if img small enough
    if curr_pixels <= max_pixels:
        return img

    # scaling factor
    scale = (max_pixels / curr_pixels) ** 0.5

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def quantize2images(images: list[str], l_bins: int, a_bins: int, b_bins: int, normalization: str, adjusted_bin_size: bool, weights: list[float]=[1.0, 1.0]) -> NDArray:
    """Creates combined histogram of 2 input images.
    
    Args:
        filepaths (list(str)): List containing 2 paths to chosen image files
        l_bins (int): Amount of bins on lightness-channel
        a_bins (int): Amount of bins on a-channel
        b_bins (int): Amount of bins on b-channel
        normalization (str): Type of normalization to apply ('L1' or 'L2')
        adjusted_bin_size (bool): Whether to adjust the bin size based on the image content. Normalization must be 'L1'
        weights (list[float]): Weights for each image's histogram contribution. Sum must be 2 and each weight must be positive.

    Returns:
        histogram_vector (NDArray): Normalized combined histogram vector of quantized colors in origian CIE-LAB 
    """
    if len(images) == 1:
        raise ValueError(f"Invalid amount of image paths in argument 'filepaths'. Expected 2, got 1. Did you mean 'quantized_image()'?")
    elif len(images) != 2:
        raise ValueError(f"Invalid amount of image paths in argument 'filepaths'. Expected 2, got {len(images)}")

    assert np.sum(weights) == 2, f"Weights must sum to 2. Current sum: {np.sum(weights)}"
    assert not (normalization != 'L1' and adjusted_bin_size), f"Adjusted bin size can only be used with normalization 'L1', not '{normalization}'"

    # Check how weights are balanced and skip unnecessary computations
    if float(weights[0]) == 0.0:
        return quantized_image(images[1], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=normalization, adjusted_bin_size=adjusted_bin_size)

    elif float(weights[1]) == 0.0:   # second weight == 0: calculate only first
        return quantized_image(images[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=normalization, adjusted_bin_size=adjusted_bin_size)

    # Both weights are non-zero, proceed with full calculation
    # Calc individual hists
    hist_1 = quantized_image(images[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=normalization, adjusted_bin_size=adjusted_bin_size)*float(weights[0])
    hist_2 = quantized_image(images[1], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=normalization, adjusted_bin_size=adjusted_bin_size)*float(weights[1])
    combined_hist = np.sum([hist_1, hist_2], axis=0)

    # Normalize sum as well
    if normalization == 'L1':
        combined_hist /= np.sum(combined_hist)
    else:   
        # L2 normalization
        combined_hist /= max(np.linalg.norm(combined_hist), 1e-12)

    return combined_hist