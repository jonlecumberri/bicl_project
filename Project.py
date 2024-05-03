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
plt.axis("off")
plt.show()


def images_to_gray_scale(ims):
    images_gray = {}
    i = 1
    for filename, image in ims.items():
        if i % 2 != 0:
            im_g = images[filename][:,:,0]
            images_gray[filename] = im_g / im_g.max()
        i = i + 1 
    return images_gray
        
images_gray = images_to_gray_scale(images)


im_001_gray = images_gray["001.bmp"]
plt.imshow(im_001_gray, cmap = "gray")
plt.axis("off")
plt.show()


#%%
# =============================================================================
# Threshold estimation for nucleus segmentation
# =============================================================================

maxIm = im_001_gray.max()
minIm = im_001_gray.min()


n1 = np.linspace(1,100,100)
n2 = 3
n2_2 = minIm * n1 / maxIm
n2_3 = minIm / (1 - maxIm/n1)

sensN1 =  (maxIm) / (maxIm + minIm * n1/n2_3)

plt.plot(n1[0:10], -sensN1[0:10])
plt.xlabel("n1")
plt.ylabel("Υn1")
plt.title("Sensivity for image 1")
plt.show()

for n in images_gray:
    im = images_gray[n]
    
    maxIm = im.max()
    minIm = im.min()
    print(maxIm, minIm, "\n")
    
    n1 = np.linspace(1,100,100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n2 = minIm / (1 - maxIm/n1)
    
    sensN1 =  (maxIm) / (maxIm + minIm * n1/n2)
    
    plt.plot(n1[0:10], -sensN1[0:10])
 
plt.xlabel("n1")
plt.ylabel("Υn1")
plt.show()


for n in images_gray:
    im = images_gray[n]
    
    maxIm = im.max()
    minIm = im.min()
    
    n2 = np.linspace(1,100,100)
    #n2 = 3
    #n2_2 = minIm * n1 / maxIm
    n1 = maxIm / (1 - minIm/n2)
    
    sensN2 =  (minIm) / (minIm + maxIm * n2/n1)
    
    plt.plot(n2[0:10], -sensN2[0:10])
 
plt.xlabel("n2")
plt.ylabel("Υn2")
plt.show()

















