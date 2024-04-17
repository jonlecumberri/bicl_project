# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:05:51 2024

@author: jla
"""

import os
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Retrieval of images
# =============================================================================


cwd = os.getcwd()
folder_images = "Dataset 2"
folder_image_path = os.path.join(cwd, folder_images)


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            image = plt.imread(path)
            if image is not None:
                images[filename] = image
    return images

images = load_images_from_folder(folder_image_path)

im_001 = images["001.bmp"]

plt.imshow(im_001[:,:,2])

def images_to_gray_scale(ims):
    images_gray = {}
    i = 1
    for filename, image in ims.items():
        if i % 2 != 0:
            im_g = images[filename][:,:,0]
            images_gray[filename] = im_g
        i = i + 1 
    return images_gray
        
images_gray = images_to_gray_scale(images)