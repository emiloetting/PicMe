import os
import sys
import annoy as ann
import numpy as np
from tqdm import tqdm




l2_vecs = np.load(os.path.join(os.getcwd(), 'ColorSimilarity', 'FullHists_L2.npy')) 
path_l2 = os.path.join(os.getcwd(), 'ColorSimilarity', 'color_index_l2.ann')


def build_index(path_to_store, vecs):
    '''Builds an Annoy index for the given vectors and stores it at the specified path.
    
    Args:
        path_to_store (str): The path where the Annoy index will be stored.
        vecs (np.ndarray): The color vectors to index.
    '''
    trees = 40
    dim = vecs.shape[1]     # adapt to size of vectors

    color_index = ann.AnnoyIndex(dim, 'angular')    # angular for euclidean -> cosine with L2 normalized
    with tqdm(total=len(vecs), desc="Building index", unit="item") as pbar:
        for i, color in enumerate(vecs):
            color_index.add_item(i, color)
            pbar.update(1)

    color_index.build(trees)
    color_index.save(path_to_store)

    return




if __name__ == '__main__':
    sys.exit(build_index(path_l2, l2_vecs))

