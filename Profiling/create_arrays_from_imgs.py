import os
import numpy as np
import cv2 as cv


cwd = os.getcwd()
img_path_1 = 

def load_and_save(img_path, dst_path):
    img = cv.imread(img_path)
    if img is not None:
        np.save(dst_path, img)
    else:
        print(f"Error loading image from {img_path}")

