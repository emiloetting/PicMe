import os
from PIL import Image
from annoy import AnnoyIndex
import json
import clip
import torch
import time


def get_image_embedding(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).squeeze().cpu().numpy()
        return embedding
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def get_best_images(image_path: str, index_to_path_json: str, num_results: int = 5):

    index = AnnoyIndex(512, 'angular')
    index.load(r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\500k.ann")

    with open(index_to_path_json, 'r') as f:
        index_to_path = json.load(f)

    embedding = get_image_embedding(image_path=image_path)
    similar_indices = index.get_nns_by_vector(embedding, num_results)
    
    similar_image_paths = []
    for idx in similar_indices:
        similar_image_paths.append(index_to_path[str(idx)])
    
    return similar_image_paths

 


if __name__ == "__main__":
    start_time = time.time()
    results = get_best_images(image_path = r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\test\hummingbird-2139278_1920.jpg", 
                   index_to_path_json=r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\test_pictures_paths.json")
    print(f"Time to find similar images: {time.time() - start_time}")
    print(results)