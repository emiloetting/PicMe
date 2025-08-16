from SSIM.ssim import *
import os

if __name__ == '__main__':
    cwd = os.getcwd()
    db = os.path.join(cwd, "SSIM", "hash_database.db")
    input_images = [os.path.join(cwd, "Profiling", "test_img_1.jpg"), os.path.join(cwd, "Profiling", "test_img_2.jpg")]   
    
    
    if len(input_images) > 1:
        results_multi = get_ssim(input_images, db_path=db)


