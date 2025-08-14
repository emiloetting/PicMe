import os
from tqdm import tqdm
import numpy as np
from skimage.color import deltaE_ciede2000
import sys
import cv2



# Determine bin sizes for embeddings
L_BINS = 5
A_BINS = 15
B_BINS = 15



def quantized_image(filepath: str, l_bins: int, a_bins: int, b_bins: int, normalization: str|None, adjusted_bin_size: bool=False):
    """
    Function to extract color signature from an image and map it to quantized LAB colors.

    Args:
        filepath (str): Path to the image file
        l_bins (int): Amount of bins on lightness-channel
        a_bins (int): Amount of bins on a-channel
        b_bins (int): Amount of bins on b-channel
        normalization (str): Type of normalization to apply ('L1' or 'L2')
        adjusted_bin_size (bool): Whether to adjust bin sizes for faster EMD calculation when normalization is 'L1'

    Returns:
        histogram_vector (NDArray): Normalized histogram vector of quantized colors in origian CIE-LAB 
    """
    assert not (normalization != 'L1' and adjusted_bin_size), f"Adjusted bin size can only be used with normalization 'L1', not '{normalization}'"
    # Check if the normalization parameter is valid
    if normalization not in ['L1', 'L2', None]:
        raise ValueError(f"Normalization must be either 'L1' or 'L2', not '{normalization}'")

    img = cv2.imread(filepath)
    rsz_img = downscale_to_fix_size(img=img, max_pixels=1_000_000)
    lab_img = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2Lab)

    # Median-Filter auf a und b
    # l, a, b = cv2.split(lab_img)
    # a = cv2.medianBlur(a, ksize=3)
    # b = cv2.medianBlur(b, ksize=3)

    # In echten CIE-LAB Farbraum konvertieren
    # cie_lab = cv_to_cie(cv2.merge([l, a, b]))

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
    l_edges = np.linspace(0, 100, l_bins + 1)
    a_edges = np.linspace(-128, 127, a_bins + 1)
    b_edges = np.linspace(-128, 127, b_bins + 1)

    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    a_centers = (a_edges[:-1] + a_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2

    grid = np.array(np.meshgrid(l_centers, a_centers, b_centers, indexing='ij'))
    return grid.reshape(3, -1).T


def cv_to_cie(img):
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


def downscale_to_fix_size(img, max_pixels: int = 250_000):
    """
    Scales image to fixed size of max_pixels while keeping aspect ratio.
    
    Args:
        img (NDArray): input image to be scaled
        max_pixels (int): max amount of pixel for image to hold after scaling

    Returns:
        NDArray: possibly downscaled image
    """
    h, w = img.shape[:2]
    curr_pixels = h * w

    # Skip if img small enough
    if curr_pixels <= max_pixels:
        return img

    # scaling factor
    scale = (max_pixels / curr_pixels) ** 0.5

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':

    cwd = os.getcwd()
    image_data_root =  r"D:\data"    # Place image dir here

    # Collect all image paths 
    image_paths = [os.path.join(root, f) for root, dirs, files in os.walk(image_data_root) for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.gif', '.bmp'))]

    # Setup list to store embeddings
    l1_embeddings_adjusted = []
    l1_embeddings_unadjusted = []
    l2_embeddings = []
    img_sizes = []

    with tqdm(image_paths, total=len(image_paths), desc="Processing images") as bar:
        for file_path in bar:
            try:
                # Calculate histograms
                hist_l1_adjusted = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1', adjusted_bin_size=True)
                hist_l1_unadjusted = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L1', adjusted_bin_size=False)
                hist_l2 = quantized_image(file_path, l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS, normalization='L2')

                # Calc img size
                img = cv2.imread(file_path)
                img_sizes.append(img.shape[:2])

                # Append to lists
                l1_embeddings_adjusted.append(hist_l1_adjusted)
                l1_embeddings_unadjusted.append(hist_l1_unadjusted)
                l2_embeddings.append(hist_l2)
                bar.set_description(f"Processing {file_path}")
            
            except:
                pass  # Skip images that cause errors
            bar.update(1)

    # Convert to arrays
    l1_embeddings_adjusted = np.array(l1_embeddings_adjusted)
    l1_embeddings_unadjusted = np.array(l1_embeddings_unadjusted)
    l2_embeddings = np.array(l2_embeddings)
    image_paths = np.array(image_paths)

    # Save embeddings (where skript is executed on GDX for simplicity)
    np.save("color_embeddings_adjusted.npy", l1_embeddings_adjusted)
    np.save("color_embeddings_unadjusted.npy", l1_embeddings_unadjusted)
    np.save("color_embeddings_l2.npy", l2_embeddings)
    np.save("image_paths.npy", image_paths)


    # ---Create EMD Cost-Matrix-----------------------------------------------------------------------------
    # Get bins in CIE-range
    center_clrs = get_bin_centers(l_bins=L_BINS, a_bins=A_BINS, b_bins=B_BINS)
    center_clrs = center_clrs.reshape(-1, 1, 3) # reshape

    # Make explicit col-vector and row-vector to allow for vectorization 
    lab1 = center_clrs[:, np.newaxis, :]
    lab2 = center_clrs[np.newaxis, :, :]

    # Calc full cost-matrix
    print("Now calculating cost-matrix using Delta E 2000 metric")
    cost_matrix = deltaE_ciede2000(lab1, lab2).astype(np.float64)
    cost_matrix = np.squeeze(cost_matrix)   # get from shape (n,n,1) to (n,n) -> required format for EMD-calc

    # Save matrix
    np.save(cost_matrix)


    sys.exit()