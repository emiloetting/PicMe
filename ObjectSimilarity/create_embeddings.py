import json
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex
from pathlib import Path
import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def image_embeddings_with_paths(folder_path):
    """
    creates clip embeddings for all images in folder

    Args:
        folder_path (str): path to folder containing images

    Yields:
        tuple: (image_path, embedding)
    """

    folder = Path(folder_path)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    
    for pattern in image_extensions:
        for image_path in tqdm.tqdm(folder.rglob(pattern)):
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(image).squeeze().cpu().numpy()
                yield str(image_path), embedding
            except Exception as e:
                print(f"Error processing {image_path}: {e}")


def create_ann(folder_path):
    """
    creates annoy index and json file mapping index to image path
    
    Args:
        folder_path (str): path to folder containing images

    Returns:
        dict: mapping of index to image path
    """
    
    index = AnnoyIndex(512, 'angular')
    path_mapping = {}
    
    for i, (image_path, embedding) in tqdm.tqdm(enumerate(image_embeddings_with_paths(folder_path))):
        try:
            index.add_item(i, embedding)
            path_mapping[i] = image_path
        except Exception as e:
            print(f"Error adding {image_path} to index: {e}")
    
    index.build(20)
    index.save("500k.ann")
    
    with open("test_pictures_paths.json", "w") as f:
        json.dump(path_mapping, f)
    
    return path_mapping


# if __name__ == "__main__":
#     mapping = create_ann(folder_path)
#     print(f" {len(mapping)} images")

