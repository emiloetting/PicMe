import cv2
import numpy as np
from scipy import signal
from hash import get_similar_images
from skimage.metrics import structural_similarity as ssim
import time
import os


def get_ssim(input_image: str, db_path: str):
    """get 5 most similar images of 500 most similar hash images to a given image"""

    similar_images = get_similar_images(image_path=input_image, 
                                    max_distance=50,
                                    max_results=500,
                                    db_path=db_path)
    
    time_start = time.time()

    image1 = cv2.imread(input_image)
    image1 = cv2.resize(image1, (32, 32), interpolation=cv2.INTER_AREA)
    
    results = []

    for similar_image in similar_images:
        image2 = cv2.imread(similar_image['id'])
        image2 = cv2.resize(image2, (32, 32), interpolation=cv2.INTER_AREA)
        similarity = ssim(image1, image2, data_range=255, channel_axis=-1)
        if similarity != 1.0:
            results.append({'similarity': similarity, 'image_path': similar_image['id']})


    results.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"Time_ssim: {time.time() - time_start}")

    return results[:5]



if __name__ == '__main__':
    cwd = os.getcwd()
    input_image = os.path.join(cwd, 'SSIM/Testbilder/dog.42.jpg')
    db = os.path.join(cwd, 'Testbilder_hashes.db')

    results = get_ssim(input_image = input_image, db_path = db)
    print(f'image_path: {input_image}')
    print()
    print(results)
    
