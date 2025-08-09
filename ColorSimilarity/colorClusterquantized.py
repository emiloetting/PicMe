import cv2	
import	numpy as np
from numpy.typing import NDArray



L_BINS = 5
A_BINS = 15
B_BINS = 15



def quantized_image(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str|None) -> NDArray:
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
    if normalization not in ['L1', 'L2', None]:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")

    img = cv2.imread(filepath)
    rsz_img = downscale(img=img)
    lab_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2Lab)

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
    elif normalization == 'L2':   
        histogram_vector /= max(np.linalg.norm(histogram_vector), 1e-12)

    # If normalization == None just return unnormalized vector    
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


def downscale(img:NDArray) -> NDArray:
    "Downscales image if images reaches certain size. Magnitude of downscaling depends on image size."
    size = img.shape[:2]
    img_pxl_count = size[0] * size[1]

    size_dict = {
        # every size below 2MP: no scaling
        (4_000_000, 8_000_000): 0.7071,
        (8_000_001, float(np.inf)): 0.5,
    }

    for intervall, scale_factor in size_dict.items():

        # Check if pxl count in interval
        if (intervall[0] <= img_pxl_count <= intervall[1]):
            
            # make work on row or col-like imgs
            new_h = max(1,int(round(size[0]*scale_factor)))
            new_w = max(1,int(round(size[1]*scale_factor)))
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # resized

    # Only catches if pxl count outside defined intervals
    return img


def quantize2images(filepaths: list[str], l_bins: int, a_bins: int, b_bins: int, normalization: str) -> NDArray:
    """Creates combined histogram of 2 input images.
    
    Args:
        filepaths (list(str)): List containing 2 paths to chosen image files
        l_bins (int): Amount of bins on lightness-channel
        a_bins (int): Amount of bins on a-channel
        b_bins (int): Amount of bins on b-channel
        normalization (str): Type of normalization to apply ('L1' or 'L2')

    Returns:
        histogram_vector (NDArray): Normalized combined histogram vector of quantized colors in origian CIE-LAB 
    """
    if len(filepaths) == 1:
        raise ValueError(f"Invalid amount of image paths in argument 'filepaths'. Expected 2, got 1. Did you mean 'quantized_image()'?")
    elif len(filepaths) != 2:
        raise ValueError(f"Invalid amount of image paths in argument 'filepaths'. Expected 2, got {len(filepaths)}")

    # Calc individual hists
    hist_1 = quantized_image(filepaths[0], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=None)
    hist_2 = quantized_image(filepaths[1], l_bins=l_bins, a_bins=a_bins, b_bins=b_bins, normalization=None)
    combined_hist = np.sum([hist_1, hist_2], axis=0)

    # Normalize sum as well
    if normalization == 'L1':
        combined_hist /= np.sum(combined_hist)
    else:   
        # L2 normalization
        combined_hist /= max(np.linalg.norm(combined_hist), 1e-12)

    return combined_hist