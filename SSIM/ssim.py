import cv2
import numpy as np
from scipy import signal
from SSIM.hash import get_similar_images
from skimage.metrics import structural_similarity as ssim
import time
import os
import sqlite3
import pickle
from typing import List, Union


def get_ssim(input_images: Union[str, List[str]], db_path: str):
    """
    Get 5 most similar images for one or multiple input images
    Optimized version using union of candidates for better coverage
    
    Args:
        input_images: Single image path (str) or list of image paths
        db_path: Path to database
    
    Returns:
        List of dictionaries with similarity scores and metadata
    """
    # input images to list
    if isinstance(input_images, str):
        input_images = [input_images]
    
    # if single image: use single image function
    if len(input_images) == 1:
        return get_ssim_single(input_images[0], db_path)
    
    # if more than one image: use multiple image function
    else:
        return get_ssim_multiple(input_images, db_path)


def get_ssim_single(input_image: str, db_path: str):
    """get best 5 similar images for one input image"""
    similar_images = get_similar_images(image_path=input_image, 
                                    max_distance=50,
                                    max_results=2000,
                                    db_path=db_path)

    print(f"Found {len(similar_images)} similar images for hash matching")

    image1 = cv2.imread(input_image)
    image1 = cv2.resize(image1, (32, 32), interpolation=cv2.INTER_AREA)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    ids = [img['id'] for img in similar_images]
    id_placeholders = ','.join(['?'] * len(ids))
    
    cursor.execute(f'''
        SELECT id, image_32x32, image_path FROM whole_db 
        WHERE id IN ({id_placeholders})
    ''', ids)
    
    pickle_data = cursor.fetchall()
    conn.close()

    results = []
    id_to_pickle = {row[0]: row[1] for row in pickle_data}
    id_to_path = {row[0]: row[2] for row in pickle_data}  
    
    for similar_image in similar_images:
        image_id = similar_image['id']
        pickle_bytes = id_to_pickle.get(image_id)
        
        if pickle_bytes is None:
            continue
            
        try:
            image2 = pickle.loads(pickle_bytes)
            similarity = ssim(image1, image2, win_size=3)
            
            results.append({
                'similarity': similarity, 
                'image_path': id_to_path.get(image_id),
            })
                
        except Exception as e:
            print(f"cannot load image {image_id}: {e}")
            continue

    results.sort(key = lambda x: x['similarity'], reverse=True)
    final_results = results[:12]
    final_results = [result['image_path'] for result in final_results]
    print(final_results)
    return final_results


def get_ssim_multiple(input_images: List[str], db_path: str):
    """
    differnces to get_ssim_single:
    - get hash candidates from all input images
    - less candidates per input image for performance
    - calc SSIM between candidates and input images
    """
    
    # get hash candidates from all input images
    all_candidates = {} 
    
    for i, input_image in enumerate(input_images):
        similar_images = get_similar_images(image_path=input_image, 
                                        max_distance=45,  
                                        max_results=1200,  # less candidates per image
                                        db_path=db_path)
        
        
        for img_info in similar_images:
            img_id = img_info['id']
            distance = img_info['distance']
            
            if img_id not in all_candidates:
                all_candidates[img_id] = {
                    'min_distance': distance,
                    'distances': [float('inf')] * len(input_images),
                    'source_images': []
                }
            
            # update min distance and source images
            if distance < all_candidates[img_id]['distances'][i]:
                all_candidates[img_id]['distances'][i] = distance
                all_candidates[img_id]['min_distance'] = min(all_candidates[img_id]['min_distance'], distance)
                if i not in all_candidates[img_id]['source_images']:
                    all_candidates[img_id]['source_images'].append(i)
        
    # get best candidates from all input images and sort by hash distance
    candidate_items = list(all_candidates.items())
    candidate_items.sort(key=lambda x: x[1]['min_distance'])
    
    max_candidates = 2500
    if len(candidate_items) > max_candidates:
        candidate_items = candidate_items[:max_candidates]
    
    selected_candidates = dict(candidate_items)
    
    # load input images
    input_images_cv = []
    for i, img_path in enumerate(input_images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Can not not load image {img_path}")
            continue
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        input_images_cv.append(img)
    
    if len(input_images_cv) != len(input_images):
        print("Warning: Some input images could not be loaded")
        return []
    
    # get best candidates from db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    candidate_ids = list(selected_candidates.keys())
    id_placeholders = ','.join(['?'] * len(candidate_ids))
    
    cursor.execute(f'''
        SELECT id, image_32x32, image_path FROM whole_db 
        WHERE id IN ({id_placeholders})
    ''', candidate_ids)
    
    pickle_data = cursor.fetchall()
    conn.close()
    
    
    # calc ssim and calc score for each candidate
    results = []
    id_to_pickle = {row[0]: row[1] for row in pickle_data}
    id_to_path = {row[0]: row[2] for row in pickle_data}  

    
    for img_id in candidate_ids:
        pickle_bytes = id_to_pickle.get(img_id)
        if pickle_bytes is None:
            continue
            
        try:
            candidate_image = pickle.loads(pickle_bytes)
            
            # calc ssim between candidates and all input images
            ssim_scores = []
            for j, input_img in enumerate(input_images_cv):
                similarity = ssim(input_img, candidate_image, win_size=3)
                ssim_scores.append(similarity)

            # if any ssim score is below 0.1, skip this image
            if any(score < 0.1 for score in ssim_scores):
                    continue  
            
            # avg ssim score of all input images
            avg_score = np.mean(ssim_scores)
            
            # if image has more than one source image, add coverage bonus
            source_count = len(selected_candidates[img_id]['source_images'])
            coverage_bonus = min(0.05 * (source_count - 1), 0.1)  
            
            # combined score is avg ssim score plus coverage bonus
            combined_score = avg_score + coverage_bonus
            
            results.append({
                'similarity': combined_score,
                'image_path': id_to_path.get(img_id),
            })
                
        except Exception as e:
            print(f"cannot load image {img_id}: {e}")
            continue

    # sort by similarity and return top 5
    results.sort(key=lambda x: x['similarity'], reverse=True)
    final_results = results[:12]
    final_results = [result['image_path'] for result in final_results]
    print(final_results)
    return final_results

if __name__ == '__main__':
    time_start = time.time()
    cwd = os.getcwd()
    db = os.path.join(cwd, '500k3.db')


    single_image = r"E:\data\image_data\500k\pixabay_dataset_v1\images_07\3d-model-world-earth-geography-2895712.jpg"
    
    results_single = get_ssim(single_image, db_path=db)
    for i, result in enumerate(results_single):
        print(f"  {i+1}. ID {result['image_id']}: {result['similarity']:.4f}, {result['image_path']}")
    
    
    input_images = [
        r"E:\data\image_data\500k\pixabay_dataset_v1\images_07\raindrops-ripples-water-rain-4332152.jpg",
        r"E:\data\image_data\500k\pixabay_dataset_v1\images_07\rain-drops-rainy-wet-droplets-3915684.jpg"
    ]
    
    
    if len(input_images) > 1:
        results_multi = get_ssim(input_images, db_path=db)

        for i, result in enumerate(results_multi):
            print(f"{i+1}. ID {result['image_id']}:")
            print(f"Combined Score: {result['similarity']:.4f}")
            print(f"Individual SSIM: {[f'{s:.4f}' for s in result['individual_similarities']]}")
            print(f"Image Path: {result['image_path']}")
            print()
    
    time_end = time.time()
    print(f"Total execution time: {time_end - time_start:.2f} seconds")