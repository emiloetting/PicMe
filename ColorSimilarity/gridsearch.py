import os
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm
import HistComp

color_spaces = ['BGR', 'RGB', 'HSV', 'LAB', 'LUV', 'HLS']
metrics = ['CORREL', 'INTERSECT', 'CHISQR', 'HELLINGER']

cwd = os.getcwd()
img_folder_path = os.path.join(cwd, 'ColorSimilarity/demo_images')
image_paths = [os.path.join(img_folder_path, image) for image in os.listdir(img_folder_path) if os.path.isfile(os.path.join(img_folder_path, image))] 

eval_filename = 'grid_results.txt'
eval_filepath = os.path.join(cwd, 'ColorSimilarity', eval_filename)

amt_comparisons = len(image_paths)*len(image_paths)*len(color_spaces)*len(metrics)
tqdm_bar = '[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'

def grid_search(img_paths: list, color_spaces: list, metrics: list):
    with open(eval_filepath, 'w') as file:
        file.write('=== Grid-Search Results for Color Space and Metric ===\n\n')
        file.write('This is the result of executing "gridsearch.py" as __main__!')
    
        with tqdm(total=amt_comparisons, desc="Image Comparisons", bar_format=tqdm_bar, colour='red') as progress_bar:
            for image1 in img_paths:
                for image2 in image_paths:
                    for color_space in color_spaces:
                        for metric in metrics:
                            img1 = cv.imread(image1)
                            img2 = cv.imread(image2)

                            folders, image1_name = os.path.split(image1)
                            folders, image2_name = os.path.split(image2)
                            file.write(f'Comparing image {image1_name} to {image2_name}\n')
                            file.write(f'Color Space: {color_space}\n')
                            file.write(f'Metric: {metric}\n')
                            img1_hists = HistComp.get_histograms(img=img1, hist_color_space=color_space)    # returns already normalized values in range [0,1]
                            img2_hists = HistComp.get_histograms(img=img2, hist_color_space=color_space)

                            channel_similarity_vector = HistComp.get_similarity(histograms=[img1_hists, img2_hists], method=metric)
                            file.write(f'Channel similarity Vector: {channel_similarity_vector}\n')
                            float_similarity = HistComp.distance_as_float(channel_similarity_vector)

                            if metric in ['CORREL', 'INTERSECT']:
                                file.write('Following float is to be MAXIMIZED for High Similarity:\n')
                            elif metric in ['CHISQR', 'HELLINGER']:
                                file.write('Following float is to be MINIMIZED for High Similarity:\n')

                            file.write(f'Similarity: {float_similarity}\n\n')
                            progress_bar.update(1)



if __name__ == '__main__':
    grid_search(img_paths=image_paths, color_spaces=color_spaces, metrics=metrics)
    sys.exit()

        
                    