import os
from PIL import Image
from annoy import AnnoyIndex
from create_embeddings import get_image_embedding


def get_best_images(image_path: str, folder_path: str):

    index = AnnoyIndex(512, 'angular')
    index.load("test_pictures.ann")

    Image.open(image_path)
    embedding = get_image_embedding(image_path = image_path)
    index = index.get_nns_by_vector(embedding, 5)

    best_images = []

    for i in index:
        best_images.append(os.listdir(folder_path)[index[i]])

    for img_path in best_images:
        Image.open(os.path.join(folder_path, img_path))

 


if __name__ == "__main__":
    print(os.path.exists(r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\pexels-photo-106685.jpeg"))
    Image.open(r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\pexels-photo-106685.jpeg")
    get_best_images(image_path = r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\test\hummingbird-2139278_1920.jpg", 
                   folder_path = r"C:\Users\joche\Documents\BigData\Repo\PicMe\ObjectSimilarity\Testbilder")
