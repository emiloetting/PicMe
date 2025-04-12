import os
import sys
import cv2 as cv
import csv
from tqdm import tqdm
import HistComp



color_spaces = ['BGR', 'RGB', 'HSV', 'LAB', 'LUV', 'HLS']
metrics = ['CORREL', 'INTERSECT', 'CHISQR', 'HELLINGER']

cwd = os.getcwd()
img_folder_path = os.path.join(cwd, 'ColorSimilarity/demo_images')
image_paths = [os.path.join(img_folder_path, image) for image in os.listdir(img_folder_path) if os.path.isfile(os.path.join(img_folder_path, image))] 

eval_filename = 'grid_results.csv'
eval_filepath = os.path.join(cwd, 'ColorSimilarity', eval_filename)

amt_comparisons = len(image_paths)*len(image_paths)*len(color_spaces)*len(metrics)
tqdm_bar = '[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'

def grid_search(img_paths: list, color_spaces: list, metrics: list):

    # Create .CSV file for evaluation
    with open(eval_filepath, 'w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['Image1', 'Image2', 'ColorSpace', 'Metric', 'SimilarityVector', 'FloatSimilarity', 'MinOrMax'])
    
        with tqdm(total=amt_comparisons, desc="Image Comparisons", bar_format=tqdm_bar, colour='red') as progress_bar:
            for image1 in img_paths:
                for image2 in image_paths:
                    for color_space in color_spaces:
                        for metric in metrics:
                            img1 = cv.imread(image1)
                            img2 = cv.imread(image2)

                            folders, image1_name = os.path.split(image1)
                            folders, image2_name = os.path.split(image2)
                            
                            csv_row = [image1_name, image2_name, color_space, metric]

                            img1_hists = HistComp.get_histograms(img=img1, hist_color_space=color_space)    # returns already normalized values in range [0,1]
                            img2_hists = HistComp.get_histograms(img=img2, hist_color_space=color_space)

                            channel_similarity_vector = HistComp.get_similarity(histograms=[img1_hists, img2_hists], method=metric)
                            float_similarity = HistComp.distance_as_float(channel_similarity_vector)
                            csv_row.append(channel_similarity_vector)
                            csv_row.append(float_similarity)

                            if metric in ['CORREL', 'INTERSECT']:
                                csv_row.append('maximize')
                            elif metric in ['CHISQR', 'HELLINGER']:
                                csv_row.append('minimize')

                            file_writer.writerow(csv_row)
                            progress_bar.update(1)



if __name__ == '__main__':
    grid_search(img_paths=image_paths, color_spaces=color_spaces, metrics=metrics)
    sys.exit()

        
                    