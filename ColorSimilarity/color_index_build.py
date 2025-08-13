import annoy as ann
import os
import numpy as np
from tqdm import tqdm




l2_vecs = np.load(os.path.join(os.getcwd(), 'ColorSimilarity', 'FullHists_L2.npy')) 
path_l2 = os.path.join(os.getcwd(), 'ColorSimilarity', 'color_index_l2.ann')


def build_index(path_to_store, vecs):
    # Create an Annoy index for the color data
    trees = 40
    dim = vecs.shape[1]     # adapt to size of vectors

    color_index = ann.AnnoyIndex(dim, 'angular')    # angular for euclidean -> cosine with L2 normalized
    with tqdm(total=len(vecs), desc="Building index", unit="item") as pbar:
        for i, color in enumerate(vecs):
            color_index.add_item(i, color)
            pbar.update(1)

    color_index.build(trees)
    color_index.save(path_to_store)




if __name__ == '__main__':
    build_index(path_l2, l2_vecs)
