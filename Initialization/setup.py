import os 
import sys
import gdown


# --------Skript to set up and download all required files for user to run the project---------------------------

if __name__ == "__main__":

    cwd = os.getcwd()


    # Define address of Google Drive folder to grab ANNOY-Index 
    color_index_file_id = None  # replace with actual ANN-index name
    ssim_index_file_id = None  # replace with actual SSIM index name
    db_path_file_id = None  # replace with actual path to database
    emd_cost_matrix_file_id = None  # replace with actual EMD cost matrix name

    # Define output paths
    color_index_output = os.path.join(cwd, "ColorSimilarity", "color_index_l2.ann")  # replace with actual output directory
    ssim_index_output = os.path.join(cwd, "ObjectSimilarity", "ssim_index_l2.ann")  # replace with actual output directory
    db_path_output = os.path.join(cwd, "Database", "database.db")  # replace with actual output directory
    emd_cost_matrix_output = os.path.join(cwd, "ColorSimilarity", "CostMatrix_full2.ann")  # replace with actual output directory


    # Make sure destination directories exist
    os.makedirs(os.path.dirname(color_index_output), exist_ok=True)
    os.makedirs(os.path.dirname(ssim_index_output), exist_ok=True)
    os.makedirs(os.path.dirname(db_path_output), exist_ok=True)
    os.makedirs(os.path.dirname(emd_cost_matrix_output), exist_ok=True)


    # Check whether files already exist
    if os.path.exists(color_index_output):
        print(f"{color_index_output} already exists.")
    if os.path.exists(ssim_index_output):
        print(f"{ssim_index_output} already exists.")
    if os.path.exists(db_path_output):
        print(f"{db_path_output} already exists.")
    if os.path.exists(emd_cost_matrix_output):
        print(f"{emd_cost_matrix_output} already exists.")


    # Actual download process
    if not os.path.exists(color_index_output):
        gdown.download(f"https://drive.google.com/uc?id={color_index_file_id}", color_index_output, quiet=False)

    if not os.path.exists(ssim_index_output):
        gdown.download(f"https://drive.google.com/uc?id={ssim_index_file_id}", ssim_index_output, quiet=False)

    if not os.path.exists(db_path_output):
        gdown.download(f"https://drive.google.com/uc?id={db_path_file_id}", db_path_output, quiet=False)

    if not os.path.exists(emd_cost_matrix_output):
        gdown.download(f"https://drive.google.com/uc?id={emd_cost_matrix_file_id}", emd_cost_matrix_output, quiet=False)

    sys.exit()
