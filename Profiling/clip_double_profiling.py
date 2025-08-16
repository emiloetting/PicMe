from ObjectSimilarity.similar_image  import *
import os
if __name__ == "__main__":
    cwd = os.getcwd()
    input_images = [os.path.join(cwd, "Profiling", "test_img_1.jpg"), os.path.join(cwd, "Profiling", "test_img_2.jpg")]   
    index_to_path_json = os.path.join(cwd, "ObjectSimilarity", "clip_embeddings_paths.json")
    annfile = os.path.join(cwd, "ObjectSimilarity", "clip_embeddings.ann")
    get_best_images(input_images, index_to_path_json, annfile)
