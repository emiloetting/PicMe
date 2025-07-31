import os
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_image_embedding(image_path):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float()

    embedding_flat = image_features.squeeze().numpy()
    return embedding_flat



def create_ann(folder_path: str):
    index = AnnoyIndex(512, 'angular')

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        embedding = get_image_embedding(image_path)
        index.add_item(index.get_n_items(), embedding)

    index.build(10)

    index.save("test_pictures.ann")

    return None



if __name__ == "__main__":
    create_ann(r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\Testbilder")