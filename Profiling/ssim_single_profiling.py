from SSIM.ssim import *

if __name__ == '__main__':
    cwd = os.getcwd()
    db = r"C:\Users\joche\Documents\BigData\Repo\PicMe\SSIM\500k3.db"
    input_images = [r"C:\Users\joche\Documents\BigData\Repo\PicMe\Profiling\test_img_1.jpg"]   
    
    
    results_multi = get_ssim(input_images, db_path=db)

