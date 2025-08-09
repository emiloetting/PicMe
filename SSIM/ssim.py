import cv2
import numpy as np
from scipy import signal
from hash import get_similar_images
from skimage.metrics import structural_similarity as ssim
import time
import os
import sqlite3
import pickle


def get_ssim(input_image: str, db_path: str):
    """get 5 most similar images of 500 most similar hash images to a given image"""

    similar_images = get_similar_images(image_path=input_image, 
                                    max_distance=50,
                                    max_results=20000,
                                    db_path=db_path)


    image1 = cv2.imread(input_image)
    image1 = cv2.resize(image1, (32, 32), interpolation=cv2.INTER_AREA)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    ids = [img['id'] for img in similar_images]
    id_placeholders = ','.join(['?'] * len(ids))
    
    cursor.execute(f'''
        SELECT id, image_32x32 FROM whole_db 
        WHERE id IN ({id_placeholders})
    ''', ids)
    
    pickle_data = cursor.fetchall()
    conn.close()

    results = []
    id_to_pickle = {row[0]: row[1] for row in pickle_data}
    
    for similar_image in similar_images:
        image_id = similar_image['id']
        pickle_bytes = id_to_pickle.get(image_id)
        
        if pickle_bytes is None:
            continue
            
        try:
            # load pickle
            image2 = pickle.loads(pickle_bytes)
            
            # calc ssim
            similarity = ssim(image1, image2, data_range=255, channel_axis=-1)
            
            if similarity != 1.1:
                results.append({
                    'similarity': similarity, 
                    'image_id': image_id,
                    'hash_distance': similar_image['distance']
                })
                
        except Exception as e:
            print(f"Pickle-Fehler bei ID {image_id}: {e}")
            continue

    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results[:5]


if __name__ == '__main__':
    time_start = time.time()
    cwd = os.getcwd()
    input_image = os.path.join(r"C:\Users\joche\Documents\BigData\Repo\PicMe\SSIM\Testbilder\3d-model-world-earth-geography-2894348.jpg")
    db = os.path.join(cwd, '500k.db')

    results = get_ssim(input_image = input_image, db_path = db)
    print(f'image_path: {input_image}')
    print()
    print(results)
    time_end = time.time()
    print(f"Time_ssim: {time_end - time_start}")
