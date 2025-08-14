from SSIM.ssim import *

if __name__ == '__main__':
    cwd = os.getcwd()
    db = os.path.join(cwd, 'SSIM','500k3.db')
    input_images = ["test_img_1.jpg", "test_img_2.jpg"]   
    
    
    if len(input_images) > 1:
        results_multi = get_ssim(input_images, db_path=db)

        for i, result in enumerate(results_multi):
            print(f"{i+1}. ID {result['image_id']}:")
            print(f"Combined Score: {result['similarity']:.4f}")
            print(f"Individual SSIM: {[f'{s:.4f}' for s in result['individual_similarities']]}")
            print(f"Image Path: {result['image_path']}")
            print()
    

