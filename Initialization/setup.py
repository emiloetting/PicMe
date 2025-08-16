import os 
import sys
import shutil
import gdown
from tqdm import tqdm


# --------Script to set up and download all required files for user to run the project---------------------------

if __name__ == "__main__":

    cwd = os.getcwd()

    # Define dict for file mapping
    mapping = {
        'color_ann_index.ann': os.path.join(cwd, "DataBase", "color_ann_index.ann"),
        'color_database.db': os.path.join(cwd, "DataBase", "color_database.db"),
        'emd_cost_full.npy': os.path.join(cwd, "DataBase", "emd_cost_full.npy"),
    }

    # Define folder ID
    folder_id = "1SEedjrkLqvSTt6i-mWCpVSWMxAMCV43i"

    # create temporary dir
    temp_dir = os.path.join(cwd, "temp_data")
    os.makedirs(temp_dir, exist_ok=True)

    # Load from GDrive
    gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", output=temp_dir, quiet=False, resume=True)

    # Init list with files to be deleted
    to_be_deleted = []

    # Redistribute loaded files into respective dst-dir
    with tqdm(total=len(os.listdir(temp_dir)), desc="Moving files") as pbar:
        for file in os.listdir(temp_dir):
            if file in mapping:
                src = os.path.join(temp_dir, file)
                dst = mapping[file]
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
            else:
                print(f"Calabria (destination unknown): No destination found for file <{file}> ")
                to_be_deleted.append(os.path.join(temp_dir, file))
            pbar.update(1)

    # Name all files to be deleted due to no dst dir
    if len(to_be_deleted) > 0:
        print(f"Files to be deleted:")
        for file in to_be_deleted:
            print(file)

    # Remove temp dir
    shutil.rmtree(temp_dir)

    sys.exit()