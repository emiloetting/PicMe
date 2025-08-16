from PIL import Image
from annoy import AnnoyIndex
import json
import clip
import torch
from typing import List, Union
import numpy as np
import time



def get_image_embedding(image_path: str):
    """
    creates clip embedding for image
    
    Args:
        image_path (str): path to image

    Returns:
        embedding
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).squeeze().cpu().numpy()
        return embedding
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def get_best_images(input_images: Union[str, List[str]], index_to_path_json: str, annfile: str, num_results: int = 12):

    """
    gets the best matching images from annoy index
    based on cosine similarity of clip embeddings

    Args:
        input_images (Union[str, List[str]]): image path or list of image paths
        index_to_path_json (str): path to json file mapping index to path
        annfile (str): path to annoy index file
        num_results (int, optional): number of results to return. Defaults to 5.

    Returns:
        list: list of image paths
    """
    if isinstance(input_images, str):
        input_images = [input_images]
    
    # if single image: use single image function
    if len(input_images) == 1:
        index = AnnoyIndex(512, 'angular')
        index.load(annfile)

        with open(index_to_path_json, 'r') as f:
            index_to_path = json.load(f)

        embedding = get_image_embedding(image_path=input_images[0])
        similar_indices = index.get_nns_by_vector(embedding, num_results)
        
        similar_image_paths = []
        for idx in similar_indices:
            similar_image_paths.append(index_to_path[str(idx)])
        
        return similar_image_paths
    
    # if multiple images: use mean of embeddings
    else:
        index = AnnoyIndex(512, 'angular')
        index.load(annfile)

        with open(index_to_path_json, 'r') as f:
            index_to_path = json.load(f)

        embeddings = [get_image_embedding(image_path=input_image) for input_image in input_images]
        combined_embedding = np.mean(embeddings, axis=0)
        similar_indices = index.get_nns_by_vector(combined_embedding, num_results)      
        
        similar_image_paths = []
        for idx in similar_indices:
            similar_image_paths.append(index_to_path[str(idx)])
        
        return similar_image_paths